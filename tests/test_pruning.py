#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import torch
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller

import pytest
from models import ALL_MODEL_NAMES, create_model
from apputils import SummaryGraph, onnx_name_2_pytorch_name, save_checkpoint, load_checkpoint

# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)

def find_module_by_name(model, module_to_find):
    for name, m in model.named_modules():
        if name == module_to_find:
            return m
    return None


def setup_test(arch, dataset):
    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None

    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return model, zeros_mask_dict

def test_ranked_filter_pruning():
    model, zeros_mask_dict = setup_test("resnet20_cifar", "cifar10")

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv1_p = distiller.model_find_param(model, "layer1.0.conv1.weight")
    assert conv1_p is not None

    # Test that there are no zero-channels
    assert distiller.sparsity_3D(conv1_p) == 0.0

    # Create a filter-ranking pruner
    reg_regims = {"layer1.0.conv1.weight" : [0.1, "3D"]}
    pruner = distiller.pruning.L1RankedStructureParameterPruner("filter_pruner", reg_regims)
    pruner.set_param_mask(conv1_p, "layer1.0.conv1.weight", zeros_mask_dict, meta=None)

    conv1 = find_module_by_name(model, "layer1.0.conv1")
    assert conv1 is not None
    # Test that the mask has the correct fraction of filters pruned.
    # We asked for 10%, but there are only 16 filters, so we have to settle for 1/16 filters
    expected_pruning = int(0.1 * conv1.out_channels) / conv1.out_channels
    assert distiller.sparsity_3D(zeros_mask_dict["layer1.0.conv1.weight"].mask) == expected_pruning

    # Use the mask to prune
    assert distiller.sparsity_3D(conv1_p) == 0
    zeros_mask_dict["layer1.0.conv1.weight"].apply_mask(conv1_p)
    assert distiller.sparsity_3D(conv1_p) == expected_pruning

    # Remove filters
    conv2 = find_module_by_name(model, "layer1.0.conv2")
    assert conv2 is not None
    assert conv1.out_channels == 16
    assert conv2.in_channels == 16

    # Test thinning
    distiller.remove_filters(model, zeros_mask_dict, "resnet20_cifar", "cifar10")
    assert conv1.out_channels == 15
    assert conv2.in_channels == 15


def test_arbitrary_channel_pruning():
    ARCH = "resnet20_cifar"
    DATASET = "cifar10"

    model, zeros_mask_dict = setup_test(ARCH, DATASET)

    conv2 = find_module_by_name(model, "layer1.0.conv2")
    assert conv2 is not None

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv2_p = distiller.model_find_param(model, "layer1.0.conv2.weight")
    assert conv2_p is not None

    assert conv2_p.dim() == 4
    num_filters = conv2_p.size(0)
    num_channels = conv2_p.size(1)
    kernel_height = conv2_p.size(2)
    kernel_width = conv2_p.size(3)

    channels_to_remove = [0, 2]

    # Let's build our 4D mask.
    # We start with a 1D mask of channels, with all but our specified channels set to one
    channels = torch.ones(num_channels)
    for ch in channels_to_remove:
        channels[ch] = 0

    # Now let's expand back up to a 4D mask
    mask = channels.expand(num_filters, num_channels)
    mask.unsqueeze_(-1)
    mask.unsqueeze_(-1)
    mask = mask.expand(num_filters, num_channels, kernel_height, kernel_width).contiguous()

    assert mask.shape == conv2_p.shape
    assert distiller.density_ch(mask) == (conv2.in_channels - len(channels_to_remove)) / conv2.in_channels

    # Cool, so now we have a mask for pruning our channels.
    # Use the mask to prune
    zeros_mask_dict["layer1.0.conv2.weight"].mask = mask
    zeros_mask_dict["layer1.0.conv2.weight"].apply_mask(conv2_p)
    all_channels = set([ch for ch in range(num_channels)])
    channels_removed = all_channels - set(distiller.find_nonzero_channels(conv2_p, "layer1.0.conv2.weight"))
    logger.info(channels_removed)

    # Now, let's do the actual network thinning
    distiller.remove_channels(model, zeros_mask_dict, ARCH, DATASET)
    conv1 = find_module_by_name(model, "layer1.0.conv1")
    logger.info(conv1)
    logger.info(conv2)
    assert conv1.out_channels == 14
    assert conv2.in_channels == 14
    assert conv1.weight.size(0) == 14
    assert conv2.weight.size(1) == 14
    bn1 = find_module_by_name(model, "layer1.0.bn1")
    assert bn1.running_var.size(0) == 14
    assert bn1.running_mean.size(0) == 14
    assert bn1.num_features == 14
    assert bn1.bias.size(0) == 14
    assert bn1.weight.size(0) == 14

    # Let's test saving and loading a thinned model.
    # We save 3 times, and load twice, to make sure to cover some corner cases:
    #   - Make sure that after loading, the model still has hold of the thinning recipes
    #   - Make sure that after a 2nd load, there no problem loading (in this case, the
    #   - tensors are already thin, so this is a new flow)
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None)
    model_2 = create_model(False, DATASET, ARCH, parallel=False)
    dummy_input = torch.randn(1, 3, 32, 32)
    model(dummy_input)
    model_2(dummy_input)
    conv2 = find_module_by_name(model_2, "layer1.0.conv2")
    assert conv2 is not None
    with pytest.raises(KeyError):
        model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')

    compression_scheduler = distiller.CompressionScheduler(model)
    hasattr(model, 'thinning_recipes')
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done")

    save_checkpoint(epoch=0, arch=ARCH, model=model_2, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    logger.info("test_arbitrary_channel_pruning - Done 2")



if __name__ == '__main__':
    test_arbitrary_channel_pruning()
