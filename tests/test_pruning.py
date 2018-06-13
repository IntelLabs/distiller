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
from apputils import SummaryGraph, onnx_name_2_pytorch_name

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

def test_ranked_filter_pruning():
    model = create_model(False, 'cifar10', 'resnet20_cifar', parallel=False)
    assert model is not None

    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker

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

    # Test masker
    assert distiller.sparsity_3D(conv1_p) == 0
    zeros_mask_dict["layer1.0.conv1.weight"].apply_mask(conv1_p)
    assert distiller.sparsity_3D(conv1_p) == expected_pruning

    # Remove filters
    assert conv1.out_channels == 16
    conv2 = find_module_by_name(model, "layer1.0.conv2")
    assert conv2 is not None
    assert conv1.in_channels == 16

    # Test thinning
    distiller.remove_filters(model, zeros_mask_dict, "resnet20_cifar", "cifar10")
    assert conv1.out_channels == 15
    assert conv2.in_channels == 15
