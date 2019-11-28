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
from collections import namedtuple
import numpy as np
import logging
import math
import torch
from functools import partial

import distiller
import common
import pytest
import inspect
from distiller.models import create_model
from distiller.apputils import save_checkpoint, load_checkpoint, load_lean_checkpoint

# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)

NetConfig = namedtuple("test_config", "arch dataset bn_name module_pairs")

# Fixture to control model data-parallelism setting
@pytest.fixture(params=[True, False])
def parallel(request):
    return request.param

#
# Model configurations
#
def simplenet(is_parallel):
    if is_parallel:
        return NetConfig(arch="simplenet_cifar", dataset="cifar10",
                         module_pairs=[("module.conv1", "module.conv2")],
                         bn_name=None)
    else:
        return NetConfig(arch="simplenet_cifar", dataset="cifar10",
                         module_pairs=[("conv1", "conv2")],
                         bn_name=None)


def resnet20_cifar(is_parallel):
    if is_parallel:
        return NetConfig(arch="resnet20_cifar", dataset="cifar10",
                         module_pairs=[("module.layer1.0.conv1", "module.layer1.0.conv2")],
                         bn_name="module.layer1.0.bn1")
    else:
        return NetConfig(arch="resnet20_cifar", dataset="cifar10",
                         module_pairs=[("layer1.0.conv1", "layer1.0.conv2")],
                         bn_name="layer1.0.bn1")


def vgg19_imagenet(is_parallel):
    if is_parallel:
        return NetConfig(arch="vgg19", dataset="imagenet",
                         module_pairs=[("features.module.0", "features.module.2"),
                                       ("features.module.21", "features.module.23"),
                                       ("features.module.23", "features.module.25"),
                                       ("features.module.25", "features.module.28"),
                                       ("features.module.28", "features.module.30"),
                                       ("features.module.30", "features.module.32"),
                                       ("features.module.32", "features.module.34")],
                         bn_name=None)
    else:
        return NetConfig(arch="vgg19", dataset="imagenet",
                         module_pairs=[("features.21", "features.23"),
                                       ("features.23", "features.25"),
                                       ("features.25", "features.28"),
                                       ("features.28", "features.30"),
                                       ("features.30", "features.32"),
                                       ("features.32", "features.34")],
                         bn_name=None)


def mobilenet_imagenet(is_parallel):
    if is_parallel:
        return NetConfig(arch="mobilenet", dataset="imagenet",
                         #module_pairs=[("module.model.0.0", "module.model.1.0")],
                         module_pairs=[("module.model.1.0", "module.model.1.3")],
                         bn_name=None)
    else:
        return NetConfig(arch="mobilenet", dataset="imagenet",
                         #module_pairs=[("model.0.0", "model.1.0")],
                         module_pairs=[("model.1.0", "model.1.3")],
                         bn_name=None)


def vgg16_cifar(is_parallel):
    if is_parallel:
        return NetConfig(arch="vgg16_cifar", dataset="cifar10",
                         module_pairs=[("features.module.0", "features.module.2")],
                         bn_name=None)
    else:
        return NetConfig(arch="vgg16_cifar", dataset="cifar10",
                         module_pairs=[("features.0", "features.2")],
                         bn_name=None)


def test_ranked_filter_pruning(parallel):
    logger.info("executing: %s (invoked by %s)" % (inspect.currentframe().f_code.co_name,
                                                   inspect.currentframe().f_back.f_code.co_name))
    ranked_filter_pruning(resnet20_cifar(parallel), ratio_to_prune=0.1, is_parallel=parallel)
    ranked_filter_pruning(resnet20_cifar(parallel), ratio_to_prune=0.5, is_parallel=parallel)
    ranked_filter_pruning(simplenet(parallel),      ratio_to_prune=0.5, is_parallel=parallel)
    ranked_filter_pruning(vgg19_imagenet(parallel), ratio_to_prune=0.1, is_parallel=parallel)
    model, zeros_mask_dict = ranked_filter_pruning(vgg19_imagenet(parallel),
                                                   ratio_to_prune=0.1,
                                                   is_parallel=parallel)
    test_vgg19_conv_fc_interface(parallel, model=model, zeros_mask_dict=zeros_mask_dict)


# todo: add a similar test for ranked channel pruning
def test_prune_all_filters(parallel):
    """Pruning all of the filteres in a weights tensor of a Convolution
    is illegal and should raise an exception.
    """
    with pytest.raises(ValueError):
        ranked_filter_pruning(resnet20_cifar(parallel), ratio_to_prune=1.0,
                              is_parallel=parallel, rounding_fn=math.ceil)
    with pytest.raises(ValueError):
        ranked_filter_pruning(resnet20_cifar(parallel), ratio_to_prune=1.0,
                              is_parallel=parallel, rounding_fn=math.floor)


def ranked_filter_pruning(config, ratio_to_prune, is_parallel, rounding_fn=math.floor):
    """Test L1 ranking and pruning of filters.
    First we rank and prune the filters of a Convolutional layer using
    a L1RankedStructureParameterPruner.  Then we physically remove the
    filters from the model (via "thining" process).
    """
    logger.info("executing: %s (invoked by %s)" % (inspect.currentframe().f_code.co_name,
                                                   inspect.currentframe().f_back.f_code.co_name))

    model, zeros_mask_dict = common.setup_test(config.arch, config.dataset, is_parallel)

    for pair in config.module_pairs:
        # Test that we can access the weights tensor of the first convolution in layer 1
        conv1_p = distiller.model_find_param(model, pair[0] + ".weight")
        assert conv1_p is not None
        num_filters = conv1_p.size(0)

        # Test that there are no zero-filters
        assert distiller.sparsity_3D(conv1_p) == 0.0

        # Create a filter-ranking pruner
        pruner = distiller.pruning.L1RankedStructureParameterPruner("filter_pruner",
                                                                    group_type="Filters",
                                                                    desired_sparsity=ratio_to_prune,
                                                                    weights=pair[0] + ".weight",
                                                                    rounding_fn=rounding_fn)
        pruner.set_param_mask(conv1_p, pair[0] + ".weight", zeros_mask_dict, meta=None)

        conv1 = common.find_module_by_name(model, pair[0])
        assert conv1 is not None
        # Test that the mask has the correct fraction of filters pruned.
        # We asked for 10%, but there are only 16 filters, so we have to settle for 1/16 filters
        expected_cnt_removed_filters = int(ratio_to_prune * conv1.out_channels)
        expected_pruning = expected_cnt_removed_filters / conv1.out_channels
        masker = zeros_mask_dict[pair[0] + ".weight"]
        assert masker is not None
        assert distiller.sparsity_3D(masker.mask) == expected_pruning

        # Use the mask to prune
        assert distiller.sparsity_3D(conv1_p) == 0
        masker.apply_mask(conv1_p)
        assert distiller.sparsity_3D(conv1_p) == expected_pruning

        # Remove filters
        conv2 = common.find_module_by_name(model, pair[1])
        assert conv2 is not None
        assert conv1.out_channels == num_filters
        assert conv2.in_channels == num_filters

    # Test thinning
    distiller.remove_filters(model, zeros_mask_dict, config.arch, config.dataset, optimizer=None)
    assert conv1.out_channels == num_filters - expected_cnt_removed_filters
    assert conv2.in_channels == num_filters - expected_cnt_removed_filters

    # Test the thinned model
    dummy_input = distiller.get_dummy_input(config.dataset, distiller.model_device(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
    run_forward_backward(model, optimizer, dummy_input)
    
    return model, zeros_mask_dict


def test_arbitrary_channel_pruning(parallel):
    arbitrary_channel_pruning(resnet20_cifar(parallel),
                              channels_to_remove=[0, 2],
                              is_parallel=parallel)
    arbitrary_channel_pruning(simplenet(parallel),
                              channels_to_remove=[0, 2],
                              is_parallel=parallel)
    arbitrary_channel_pruning(vgg19_imagenet(parallel),
                              channels_to_remove=[0, 2],
                              is_parallel=parallel)
    arbitrary_channel_pruning(vgg16_cifar(parallel),
                              channels_to_remove=[0, 2],
                              is_parallel=parallel)


def test_prune_all_channels(parallel):
    """Pruning all of the channels in a weights tensor of a Convolution
    is illegal and should raise an exception.
    """
    with pytest.raises(ValueError):
        arbitrary_channel_pruning(resnet20_cifar(parallel),
                                  channels_to_remove=[ch for ch in range(16)],
                                  is_parallel=parallel)


def test_channel_pruning_conv_bias(parallel):
    arbitrary_channel_pruning(simplenet(parallel),
                              channels_to_remove=[0, 1],
                              is_parallel=parallel)


def create_channels_mask(conv_p, channels_to_remove):
    assert conv_p.dim() == 4
    num_filters = conv_p.size(0)
    num_channels = conv_p.size(1)
    kernel_height = conv_p.size(2)
    kernel_width = conv_p.size(3)

    # Let's build our 4D mask.
    # We start with a 1D mask of channels, with all but our specified channels set to one
    channels = torch.ones(num_channels)
    for ch in channels_to_remove:
        channels[ch] = 0

    # Now let's expand back up to a 4D mask
    mask = channels.expand(num_filters, num_channels)
    mask.unsqueeze_(-1)
    mask.unsqueeze_(-1)
    mask = mask.expand(num_filters, num_channels, kernel_height, kernel_width).contiguous().cuda()

    assert mask.shape == conv_p.shape
    return mask


def run_forward_backward(model, optimizer, dummy_input):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model.train()
    output = model(dummy_input)
    target = torch.LongTensor(1).random_(2).cuda()
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def arbitrary_channel_pruning(config, channels_to_remove, is_parallel):
    """Test removal of arbitrary channels.
    The test receives a specification of channels to remove.
    Based on this specification, the channels are pruned and then physically
    removed from the model (via a "thinning" process).
    """
    model, zeros_mask_dict = common.setup_test(config.arch, config.dataset, is_parallel)

    pair = config.module_pairs[0]
    conv2 = common.find_module_by_name(model, pair[1])
    assert conv2 is not None

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv2_p = distiller.model_find_param(model, pair[1] + ".weight")
    assert conv2_p is not None

    assert conv2_p.dim() == 4
    num_channels = conv2_p.size(1)
    cnt_nnz_channels = num_channels - len(channels_to_remove)
    mask = create_channels_mask(conv2_p, channels_to_remove)
    assert distiller.density_ch(mask) == (conv2.in_channels - len(channels_to_remove)) / conv2.in_channels
    # Cool, so now we have a mask for pruning our channels.

    # Use the mask to prune
    zeros_mask_dict[pair[1] + ".weight"].mask = mask
    zeros_mask_dict[pair[1] + ".weight"].apply_mask(conv2_p)
    all_channels = set([ch for ch in range(num_channels)])
    nnz_channels = set(distiller.non_zero_channels(conv2_p))
    channels_removed = all_channels - nnz_channels
    logger.info("Channels removed {}".format(channels_removed))

    # Now, let's do the actual network thinning
    distiller.remove_channels(model, zeros_mask_dict, config.arch, config.dataset, optimizer=None)
    conv1 = common.find_module_by_name(model, pair[0])
    assert conv1
    assert conv1.out_channels == cnt_nnz_channels
    assert conv2.in_channels == cnt_nnz_channels
    assert conv1.weight.size(0) == cnt_nnz_channels
    assert conv2.weight.size(1) == cnt_nnz_channels
    if config.bn_name is not None:
        bn1 = common.find_module_by_name(model, config.bn_name)
        assert bn1.running_var.size(0) == cnt_nnz_channels
        assert bn1.running_mean.size(0) == cnt_nnz_channels
        assert bn1.num_features == cnt_nnz_channels
        assert bn1.bias.size(0) == cnt_nnz_channels
        assert bn1.weight.size(0) == cnt_nnz_channels

    dummy_input = distiller.get_dummy_input(config.dataset, distiller.model_device(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
    run_forward_backward(model, optimizer, dummy_input)

    # Let's test saving and loading a thinned model.
    # We save 3 times, and load twice, to make sure to cover some corner cases:
    #   - Make sure that after loading, the model still has hold of the thinning recipes
    #   - Make sure that after a 2nd load, there no problem loading (in this case, the
    #     tensors are already thin, so this is a new flow)
    # (1)
    save_checkpoint(epoch=0, arch=config.arch, model=model, optimizer=None)
    model_2 = create_model(False, config.dataset, config.arch, parallel=is_parallel)
    model(dummy_input)
    model_2(dummy_input)
    conv2 = common.find_module_by_name(model_2, pair[1])
    assert conv2 is not None
    model_2 = load_lean_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')

    run_forward_backward(model, optimizer, dummy_input)

    # (2)
    compression_scheduler = distiller.CompressionScheduler(model)
    save_checkpoint(epoch=0, arch=config.arch, model=model, optimizer=None, scheduler=compression_scheduler)
    model_2 = load_lean_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done")

    # (3)
    save_checkpoint(epoch=0, arch=config.arch, model=model_2, optimizer=None, scheduler=compression_scheduler)
    model_2 = load_lean_checkpoint(model_2, 'checkpoint.pth.tar')
    assert hasattr(model_2, 'thinning_recipes')
    logger.info("test_arbitrary_channel_pruning - Done 2")


def conv_fc_interface_test(arch, dataset, conv_names, fc_names, is_parallel=parallel, model=None, zeros_mask_dict=None):
    """A special case of convolution filter-pruning occurs when the next layer is
    fully-connected (linear).  This test is for this case and uses VGG16.
    """
    ratio_to_prune = 0.1
    # Choose the layer names according to the data-parallelism setting
    names_idx = 0 if not is_parallel else 1
    conv_name = conv_names[names_idx]
    fc_name = fc_names[names_idx]

    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    if model is None or zeros_mask_dict is None:
        model, zeros_mask_dict = common.setup_test(arch, dataset, is_parallel)

    # Run forward and backward passes, in order to create the gradients and optimizer params
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
    run_forward_backward(model, optimizer, dummy_input)

    conv = common.find_module_by_name(model, conv_name)
    assert conv is not None

    conv_p = distiller.model_find_param(model, conv_name + ".weight")
    assert conv_p is not None
    assert conv_p.dim() == 4

    # Create a filter-ranking pruner
    pruner = distiller.pruning.L1RankedStructureParameterPruner("filter_pruner",
                                                                group_type="Filters",
                                                                desired_sparsity=ratio_to_prune,
                                                                weights=conv_name + ".weight")
    pruner.set_param_mask(conv_p, conv_name + ".weight", zeros_mask_dict, meta=None)

    # Use the mask to prune
    masker = zeros_mask_dict[conv_name + ".weight"]
    assert masker is not None
    masker.apply_mask(conv_p)
    num_filters = conv_p.size(0)
    expected_cnt_removed_filters = int(ratio_to_prune * conv.out_channels)

    # Remove filters
    fc = common.find_module_by_name(model, fc_name)
    assert fc is not None

    # Test thinning
    fm_size = fc.in_features // conv.out_channels
    num_nnz_filters = num_filters - expected_cnt_removed_filters
    distiller.remove_filters(model, zeros_mask_dict, arch, dataset, optimizer)
    assert conv.out_channels == num_nnz_filters
    assert fc.in_features == fm_size * num_nnz_filters

    # Run again, to make sure the optimizer and gradients shapes were updated correctly
    run_forward_backward(model, optimizer, dummy_input)
    run_forward_backward(model, optimizer, dummy_input)


def test_vgg19_conv_fc_interface(is_parallel=parallel, model=None, zeros_mask_dict=None):
    conv_fc_interface_test("vgg19", "imagenet", conv_names=["features.34", "features.module.34"],
                           fc_names=["classifier.0", "classifier.0"], is_parallel=is_parallel, model=model,
                           zeros_mask_dict=zeros_mask_dict)


def test_mobilenet_conv_fc_interface(is_parallel=parallel, model=None, zeros_mask_dict=None):
    conv_fc_interface_test("mobilenet", "imagenet", conv_names=["model.13.3", "module.model.13.3"],
                           fc_names=["fc", "module.fc"], is_parallel=is_parallel, model=model,
                           zeros_mask_dict=zeros_mask_dict)


def test_magnitude_pruning():
    # Create a 4-D tensor of 1s
    a = torch.ones(3, 64, 32, 32)
    # Change one element
    a[1, 4, 17, 31] = 0.2
    # Create a masks dictionary and populate it with one ParameterMasker
    zeros_mask_dict = {}
    masker = distiller.ParameterMasker('a')
    zeros_mask_dict['a'] = masker
    # Try to use a MagnitudeParameterPruner with defining a default threshold
    with pytest.raises(AssertionError):
        pruner = distiller.pruning.MagnitudeParameterPruner("test", None)

    # Now define the default threshold
    thresholds = {"*": 0.4}
    pruner = distiller.pruning.MagnitudeParameterPruner("test", thresholds)
    assert distiller.sparsity(a) == 0
    # Create a mask for parameter 'a'
    pruner.set_param_mask(a, 'a', zeros_mask_dict, None)
    assert common.almost_equal(distiller.sparsity(zeros_mask_dict['a'].mask), 1/distiller.volume(a))

    # Let's now use the masker to prune a parameter
    masker = zeros_mask_dict['a']
    masker.apply_mask(a)
    assert common.almost_equal(distiller.sparsity(a), 1/distiller.volume(a))
    # We can use the masker on other tensors, if we want (and if they have the correct shape).
    # Remember that the mask was created already, so we're not thresholding - we are pruning
    b = torch.ones(3, 64, 32, 32)
    b[:] = 0.3
    masker.apply_mask(b)
    assert common.almost_equal(distiller.sparsity(b), 1/distiller.volume(a))


def test_row_pruning():
    param = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
    from distiller.pruning import L1RankedStructureParameterPruner

    masker = distiller.scheduler.ParameterMasker("debug name")
    zeros_mask_dict = {"some name": masker}
    L1RankedStructureParameterPruner.rank_and_prune_channels(0.5, param, "some name", zeros_mask_dict)
    print(distiller.sparsity_rows(masker.mask))
    assert math.isclose(distiller.sparsity_rows(masker.mask), 1/3)
    pass


if __name__ == '__main__':
    for is_parallel in [True, False]:
        test_ranked_filter_pruning(is_parallel)
        test_arbitrary_channel_pruning(is_parallel)
        test_prune_all_channels(is_parallel)
        model, zeros_mask_dict = ranked_filter_pruning(vgg19_imagenet(is_parallel),
                                                       ratio_to_prune=0.1,
                                                       is_parallel=is_parallel)
        test_vgg19_conv_fc_interface(is_parallel)
        arbitrary_channel_pruning(vgg19_imagenet(is_parallel),
                                  channels_to_remove=[0, 2],
                                  is_parallel=is_parallel)
        model, zeros_mask_dict = ranked_filter_pruning(mobilenet_imagenet(False),
                                                       ratio_to_prune=0.1,
                                                       is_parallel=False)
        test_mobilenet_conv_fc_interface(is_parallel)
        test_vgg19_conv_fc_interface(is_parallel)
        arbitrary_channel_pruning(mobilenet_imagenet(is_parallel),
                                  channels_to_remove=[0, 2],
                                  is_parallel=is_parallel)
    test_row_pruning()