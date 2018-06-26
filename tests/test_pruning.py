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
try:
    import distiller
except ImportError:
    module_path = os.path.abspath(os.path.join('..'))
    sys.path.append(module_path)
    import distiller
import common
import pytest
from models import create_model
from apputils import save_checkpoint, load_checkpoint

# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def test_ranked_filter_pruning():
    ranked_filter_pruning(ratio_to_prune=0.1)
    ranked_filter_pruning(ratio_to_prune=0.5)


def test_prune_all_filters():
    """Pruning all of the filteres in a weights tensor of a Convolution
    is illegal and should raise an exception.
    """
    with pytest.raises(ValueError):
        ranked_filter_pruning(ratio_to_prune=1.0)


def ranked_filter_pruning(ratio_to_prune):
    """Test L1 ranking and pruning of filters.

    First we rank and prune the filters of a Convolutional layer using
    a L1RankedStructureParameterPruner.  Then we physically remove the
    filters from the model (via "thining" process).
    """
    model, zeros_mask_dict = common.setup_test("resnet20_cifar", "cifar10")

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv1_p = distiller.model_find_param(model, "layer1.0.conv1.weight")
    assert conv1_p is not None

    # Test that there are no zero-filters
    assert distiller.sparsity_3D(conv1_p) == 0.0

    # Create a filter-ranking pruner
    reg_regims = {"layer1.0.conv1.weight": [ratio_to_prune, "3D"]}
    pruner = distiller.pruning.L1RankedStructureParameterPruner("filter_pruner", reg_regims)
    pruner.set_param_mask(conv1_p, "layer1.0.conv1.weight", zeros_mask_dict, meta=None)

    conv1 = common.find_module_by_name(model, "layer1.0.conv1")
    assert conv1 is not None
    # Test that the mask has the correct fraction of filters pruned.
    # We asked for 10%, but there are only 16 filters, so we have to settle for 1/16 filters
    expected_cnt_removed_filters = int(ratio_to_prune * conv1.out_channels)
    expected_pruning = expected_cnt_removed_filters / conv1.out_channels
    assert distiller.sparsity_3D(zeros_mask_dict["layer1.0.conv1.weight"].mask) == expected_pruning

    # Use the mask to prune
    assert distiller.sparsity_3D(conv1_p) == 0
    zeros_mask_dict["layer1.0.conv1.weight"].apply_mask(conv1_p)
    assert distiller.sparsity_3D(conv1_p) == expected_pruning

    # Remove filters
    conv2 = common.find_module_by_name(model, "layer1.0.conv2")
    assert conv2 is not None
    assert conv1.out_channels == 16
    assert conv2.in_channels == 16

    # Test thinning
    distiller.remove_filters(model, zeros_mask_dict, "resnet20_cifar", "cifar10")
    assert conv1.out_channels == 16 - expected_cnt_removed_filters
    assert conv2.in_channels == 16 - expected_cnt_removed_filters


def test_arbitrary_channel_pruning():
    arbitrary_channel_pruning(channels_to_remove=[0, 2])


def test_prune_all_channels():
    """Pruning all of the channels in a weights tensor of a Convolution
    is illegal and should raise an exception.
    """
    with pytest.raises(ValueError):
        arbitrary_channel_pruning(channels_to_remove=[ch for ch in range(16)])


def arbitrary_channel_pruning(channels_to_remove):
    """Test removal of arbitrary channels.

    The test receives a specification of channels to remove.
    Based on this specification, the channels are pruned and then physically
    removed from the model (via a "thinning" process).
    """
    ARCH = "resnet20_cifar"
    DATASET = "cifar10"

    model, zeros_mask_dict = common.setup_test(ARCH, DATASET)

    conv2 = common.find_module_by_name(model, "layer1.0.conv2")
    assert conv2 is not None

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv2_p = distiller.model_find_param(model, "layer1.0.conv2.weight")
    assert conv2_p is not None

    assert conv2_p.dim() == 4
    num_filters = conv2_p.size(0)
    num_channels = conv2_p.size(1)
    kernel_height = conv2_p.size(2)
    kernel_width = conv2_p.size(3)
    cnt_nnz_channels = num_channels - len(channels_to_remove)

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
    nnz_channels = set(distiller.find_nonzero_channels_list(conv2_p, "layer1.0.conv2.weight"))
    channels_removed = all_channels - nnz_channels
    logger.info("Channels removed {}".format(channels_removed))

    # Now, let's do the actual network thinning
    distiller.remove_channels(model, zeros_mask_dict, ARCH, DATASET)
    conv1 = common.find_module_by_name(model, "layer1.0.conv1")
    logger.info(conv1)
    logger.info(conv2)
    assert conv1.out_channels == cnt_nnz_channels
    assert conv2.in_channels == cnt_nnz_channels
    assert conv1.weight.size(0) == cnt_nnz_channels
    assert conv2.weight.size(1) == cnt_nnz_channels
    bn1 = common.find_module_by_name(model, "layer1.0.bn1")
    assert bn1.running_var.size(0) == cnt_nnz_channels
    assert bn1.running_mean.size(0) == cnt_nnz_channels
    assert bn1.num_features == cnt_nnz_channels
    assert bn1.bias.size(0) == cnt_nnz_channels
    assert bn1.weight.size(0) == cnt_nnz_channels

    # Let's test saving and loading a thinned model.
    # We save 3 times, and load twice, to make sure to cover some corner cases:
    #   - Make sure that after loading, the model still has hold of the thinning recipes
    #   - Make sure that after a 2nd load, there no problem loading (in this case, the
    #   - tensors are already thin, so this is a new flow)
    # (1)
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None)
    model_2 = create_model(False, DATASET, ARCH, parallel=False)
    dummy_input = torch.randn(1, 3, 32, 32)
    model(dummy_input)
    model_2(dummy_input)
    conv2 = common.find_module_by_name(model_2, "layer1.0.conv2")
    assert conv2 is not None
    with pytest.raises(KeyError):
        model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    compression_scheduler = distiller.CompressionScheduler(model)
    hasattr(model, 'thinning_recipes')

    # (2)
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done")

    # (3)
    save_checkpoint(epoch=0, arch=ARCH, model=model_2, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done 2")


def test_channel_pruning_conv_bias():
    """Test removal of arbitrary channels, for Convolutions with bias term.

    This is different from test_arbitrary_channel_pruning() in that this model
    has Convolution layers with biases tensors.
    """
    ARCH = "simplenet_cifar"
    DATASET = "cifar10"
    channels_to_remove = [0, 1]
    has_bn = False
    model, zeros_mask_dict = common.setup_test(ARCH, DATASET)

    conv2 = common.find_module_by_name(model, "conv2")
    assert conv2 is not None

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv2_p = distiller.model_find_param(model, "conv2.weight")
    assert conv2_p is not None

    assert conv2_p.dim() == 4
    num_filters = conv2_p.size(0)
    num_channels = conv2_p.size(1)
    kernel_height = conv2_p.size(2)
    kernel_width = conv2_p.size(3)
    cnt_nnz_channels = num_channels - len(channels_to_remove)

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
    zeros_mask_dict["conv2.weight"].mask = mask
    zeros_mask_dict["conv2.weight"].apply_mask(conv2_p)
    all_channels = set([ch for ch in range(num_channels)])
    nnz_channels = set(distiller.find_nonzero_channels_list(conv2_p, "conv2.weight"))
    channels_removed = all_channels - nnz_channels
    logger.info("Channels removed {}".format(channels_removed))

    # Now, let's do the actual network thinning
    distiller.remove_channels(model, zeros_mask_dict, ARCH, DATASET)
    conv1 = common.find_module_by_name(model, "conv1")
    logger.info(conv1)
    logger.info(conv2)
    assert conv1.out_channels == cnt_nnz_channels
    assert conv2.in_channels == cnt_nnz_channels
    assert conv1.weight.size(0) == cnt_nnz_channels
    assert conv2.weight.size(1) == cnt_nnz_channels
    if has_bn:
        bn1 = common.find_module_by_name(model, "layer1.0.bn1")
        assert bn1.running_var.size(0) == cnt_nnz_channels
        assert bn1.running_mean.size(0) == cnt_nnz_channels
        assert bn1.num_features == cnt_nnz_channels
        assert bn1.bias.size(0) == cnt_nnz_channels
        assert bn1.weight.size(0) == cnt_nnz_channels

    # Let's test saving and loading a thinned model.
    # We save 3 times, and load twice, to make sure to cover some corner cases:
    #   - Make sure that after loading, the model still has hold of the thinning recipes
    #   - Make sure that after a 2nd load, there no problem loading (in this case, the
    #   - tensors are already thin, so this is a new flow)
    # (1)
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None)
    model_2 = create_model(False, DATASET, ARCH, parallel=False)
    dummy_input = torch.randn(1, 3, 32, 32)
    model(dummy_input)
    model_2(dummy_input)
    conv2 = common.find_module_by_name(model_2, "conv2")
    assert conv2 is not None
    with pytest.raises(KeyError):
        model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    compression_scheduler = distiller.CompressionScheduler(model)
    hasattr(model, 'thinning_recipes')

    # (2)
    save_checkpoint(epoch=0, arch=ARCH, model=model, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done")

    # (3)
    save_checkpoint(epoch=0, arch=ARCH, model=model_2, optimizer=None, scheduler=compression_scheduler)
    model_2, compression_scheduler, start_epoch = load_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done 2")


if __name__ == '__main__':
    test_arbitrary_channel_pruning()
    test_prune_all_channels()
