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
import torch.nn as nn
import pytest
import os
import errno
import distiller
from distiller.models import create_model


PYTEST_COLLATERALS_DIR = os.path.join(os.path.dirname(__file__), 'pytest_collaterals')
try:
    os.makedirs(PYTEST_COLLATERALS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


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


def almost_equal(a , b, max_diff=0.000001):
    return abs(a - b) <= max_diff


def pytest_raises_wrapper(exc_type, msg, func, *args, **kwargs):
    with pytest.raises(exc_type):
        func(*args, **kwargs)
        if msg:
            pytest.fail(msg)


class WrappedSequential(nn.Module):
    def __init__(self, *args):
        super(WrappedSequential, self).__init__()
        self.seq = nn.Sequential(*args)

    def forward(self, x):
        return self.seq(x)
