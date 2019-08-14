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
import common
import distiller
import distiller.models as models


def test_sparsity():
    zeros = torch.zeros(2, 3, 5, 6)
    print(distiller.sparsity(zeros))
    assert distiller.sparsity(zeros) == 1.0
    assert distiller.sparsity_3D(zeros) == 1.0
    assert distiller.density_3D(zeros) == 0.0
    ones = torch.ones(12, 43, 4, 6)
    assert distiller.sparsity(ones) == 0.0
    x = torch.tensor([[1., 2., 0, 4., 0],
                      [1., 2., 0, 4., 0]])
    assert distiller.density(x) == 0.6
    assert distiller.density_cols(x, transposed=False) == 0.6
    assert distiller.sparsity_rows(x, transposed=False) == 0
    x = torch.tensor([[0., 0., 0],
                      [1., 4., 0],
                      [1., 2., 0],
                      [0., 0., 0]])
    assert distiller.density(x) == 4/12
    assert distiller.sparsity_rows(x, transposed=False) == 0.5
    assert common.almost_equal(distiller.sparsity_cols(x, transposed=False), 1/3)
    assert common.almost_equal(distiller.sparsity_rows(x), 1/3)


def test_activations():
    x = torch.tensor([[[[1.,  0.,  0.],
                        [0.,  2.,  0.],
                        [0.,  0.,  3.]],

                       [[1.,  0.,  2.],
                        [0.,  3.,  0.],
                        [4.,  0.,  5.]]],


                      [[[4.,  0.,  0.],
                        [0.,  5.,  0.],
                        [0.,  0.,  6.]],

                       [[0.,  6.,  0.],
                        [7.,  0.,  8.],
                        [0.,  9.,  0.]]]])
    assert all(distiller.activation_channels_l1(x) == torch.tensor([21/2,  45/2]))
    assert all(distiller.activation_channels_apoz(x) == torch.tensor([100*(6+6)/(9+9),  100*(4+5)/(9+9)]))
    assert all(distiller.activation_channels_means(x) == torch.tensor([21/18,  45/18]))


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
