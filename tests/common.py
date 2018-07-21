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

import os
import sys
import torch
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
from models import create_model


def setup_test(arch, dataset, parallel):
    model = create_model(False, dataset, arch, parallel=parallel)
    assert model is not None

    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return model, zeros_mask_dict


def find_module_by_name(model, module_to_find):
    for name, m in model.named_modules():
        if name == module_to_find:
            return m
    return None


def get_dummy_input(dataset):
    if dataset == "imagenet":
        return torch.randn(1, 3, 224, 224).cuda()
    elif dataset == "cifar10":
        return torch.randn(1, 3, 32, 32).cuda()
    raise ValueError("Trying to use an unknown dataset " + dataset)


def almost_equal(a , b, max_diff=0.000001):
    return abs(a - b) <= max_diff
