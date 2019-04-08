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
import distiller
import common  # common test code

# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def test_ch_ranking():
    # Tensor with shape [3, 2, 2, 2] -- 3 filters, 2 channels
    param = torch.tensor([[[[11., 12],
                            [13,  14]],

                           [[15., 16],
                            [17,  18]]],
                          # Filter #2
                          [[[21., 22],
                            [23,  24]],

                           [[25., 26],
                            [27,  28]]],
                          # Filter #3
                          [[[31., 32],
                            [33,  34]],

                           [[35., 36],
                            [37,  38]]]])

    fraction_to_prune = 0.5
    binary_map = distiller.pruning.L1RankedStructureParameterPruner.rank_and_prune_channels(fraction_to_prune, param)
    assert all(binary_map == torch.tensor([0.,  1.]))


def test_ranked_channel_pruning():
    model, zeros_mask_dict = common.setup_test("resnet20_cifar", "cifar10", parallel=False)

    # Test that we can access the weights tensor of the first convolution in layer 1
    conv1_p = distiller.model_find_param(model, "layer1.0.conv1.weight")
    assert conv1_p is not None

    # Test that there are no zero-channels
    assert distiller.sparsity_ch(conv1_p) == 0.0

    # # Create a channel-ranking pruner
    pruner = distiller.pruning.L1RankedStructureParameterPruner("channel_pruner",
                                                                group_type="Channels",
                                                                desired_sparsity=0.1,
                                                                weights="layer1.0.conv1.weight")
    pruner.set_param_mask(conv1_p, "layer1.0.conv1.weight", zeros_mask_dict, meta=None)

    conv1 = common.find_module_by_name(model, "layer1.0.conv1")
    assert conv1 is not None

    # Test that the mask has the correct fraction of channels pruned.
    # We asked for 10%, but there are only 16 channels, so we have to settle for 1/16 channels
    logger.info("layer1.0.conv1 = {}".format(conv1))
    expected_pruning = int(0.1 * conv1.in_channels) / conv1.in_channels
    assert distiller.sparsity_ch(zeros_mask_dict["layer1.0.conv1.weight"].mask) == expected_pruning

    # Use the mask to prune
    assert distiller.sparsity_ch(conv1_p) == 0
    zeros_mask_dict["layer1.0.conv1.weight"].apply_mask(conv1_p)
    assert distiller.sparsity_ch(conv1_p) == expected_pruning

    # Remove channels (and filters)
    conv0 = common.find_module_by_name(model, "conv1")
    assert conv0 is not None
    assert conv0.out_channels == 16
    assert conv1.in_channels == 16

    # Test thinning
    distiller.remove_channels(model, zeros_mask_dict, "resnet20_cifar", "cifar10", optimizer=None)
    assert conv0.out_channels == 15
    assert conv1.in_channels == 15
