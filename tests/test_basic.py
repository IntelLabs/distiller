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

import torch
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import models

def test_sparsity():
    zeros = torch.zeros(2,3,5,6)
    print(distiller.sparsity(zeros))
    assert distiller.sparsity(zeros) == 1.0
    assert distiller.sparsity_3D(zeros) == 1.0
    assert distiller.density_3D(zeros) == 0.0


    ones = torch.zeros(12,43,4,6)
    ones.fill_(1)
    assert distiller.sparsity(ones) == 0.0

def test_utils():
    model = models.create_model(False, 'cifar10', 'resnet20_cifar', parallel=False)
    assert model is not None

    p = distiller.model_find_param(model, "")
    assert p is None

    # Search for a parameter by its "non-parallel" name
    p = distiller.model_find_param(model, "layer1.0.conv1.weight")
    assert p is not None

    # Search for a module name
    module_to_find = None
    for name, m in model.named_modules():
        if name == "layer1.0.conv1":
            module_to_find = m
            break
    assert module_to_find is not None

    module_name = distiller.model_find_module_name(model, module_to_find)
    assert module_name == "layer1.0.conv1"
