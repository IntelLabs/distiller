#
# Copyright (c) 2019 Intel Corporation
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

import pytest
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.testing

import distiller.model_transforms as mt
from distiller.modules import EltwiseAdd, Split
from common import WrappedSequential


class BypassModel(nn.Module):
    def __init__(self, prologue, bypassed):
        super(BypassModel, self).__init__()
        self.prologue = nn.Sequential(*prologue)
        self.bypassed = bypassed
        self.add = EltwiseAdd()

    def forward(self, x):
        # t = self.b(self.a(x))
        # return self.add(t, self.d(t))
        t = self.prologue(x)
        return self.add(t, self.bypassed(t))


class Dummy(nn.Module):
    def __init__(self, fuseable=True):
        super(Dummy, self).__init__()
        self.fuseable = fuseable

    def forward(self, x):
        return F.relu(x)


class DummyA(Dummy):
    pass


class DummyB(Dummy):
    pass


class DummyC(Dummy):
    pass


class DummyD(Dummy):
    pass


def fuse_fn(sequence):
    return sequence[0]


types_sequence = (DummyA, (DummyB, DummyC), DummyD)


def fused_reference():
    return WrappedSequential(DummyA(), nn.Identity(), nn.Identity())


def fuse_and_check(model, expected, input_shape, parallel):
    if parallel:
        model = nn.DataParallel(model)
        expected = nn.DataParallel(expected)
    dummy_input = torch.randn(input_shape)
    fused = mt.fuse_modules(model, dummy_input, types_sequence=types_sequence, fuse_fn=fuse_fn)
    nm_fused = OrderedDict(fused.named_modules())
    nm_expected = OrderedDict(expected.named_modules())
    assert nm_fused.keys() == nm_expected.keys()
    assert [type(v) for v in nm_fused.values()] == [type(v) for v in nm_fused.values()]


@pytest.fixture(params=[False, True], ids=['parallel_off', 'parallel_on'])
def parallel(request):
    return request.param


def test_fuse_models(parallel):
    input_shape = (10, 10)

    # Simple negative tests

    # Not Fusable
    model = WrappedSequential(DummyA(fuseable=False), DummyB(), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyA(), DummyB(fuseable=False), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyA(), DummyC(), DummyD(fuseable=False))
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    # Wrong sequence
    model = WrappedSequential(DummyB())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyA(), DummyA(), DummyB(), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyB(), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyA(), DummyB(), DummyC(), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    # Simple positive tests

    # Simple sequence 1
    model = WrappedSequential(DummyA(), DummyB(), DummyD())
    fuse_and_check(model, fused_reference(), input_shape, parallel)

    # Simple sequence 2
    model = WrappedSequential(DummyA(), DummyC(), DummyD())
    fuse_and_check(model, fused_reference(), input_shape, parallel)

    # 2 sequences
    model = WrappedSequential(WrappedSequential(DummyA(), DummyB(), DummyD()),
                              WrappedSequential(DummyA(), DummyC(), DummyD()))
    expected = WrappedSequential(fused_reference(), fused_reference())
    fuse_and_check(model, expected, input_shape, parallel)

    # "Complex" tests

    # 2 sequences with wrong sequence between them
    model = WrappedSequential(WrappedSequential(DummyA(), DummyB(), DummyD()),
                              DummyA(), DummyB(),
                              WrappedSequential(DummyA(), DummyC(), DummyD()))
    expected = WrappedSequential(fused_reference(),
                                 DummyA(), DummyB(),
                                 fused_reference())
    fuse_and_check(model, expected, input_shape, parallel)

    # "Complex" model
    class SplitJoinModel(nn.Module):
        def __init__(self, m1, m2):
            super(SplitJoinModel, self).__init__()
            self.split = Split(int(input_shape[0] / 2))
            self.m1 = m1
            self.m2 = m2
            self.add = EltwiseAdd()

        def forward(self, x):
            # x1, x2 = self.split(x)
            y1 = self.m1(x)
            y2 = self.m2(x)
            return self.add(y1, y2)

    model = SplitJoinModel(WrappedSequential(DummyA(), DummyB(), DummyD()), WrappedSequential(DummyA(), DummyC(), DummyD()))
    expected = SplitJoinModel(fused_reference(), fused_reference())
    fuse_and_check(model, expected, input_shape, parallel)

    # Node with multiple outputs
    model = BypassModel((DummyA(), DummyB()), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)
