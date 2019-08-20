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

from distiller import SummaryGraph
import distiller.model_transforms as mt
from distiller.modules import EltwiseAdd, Split
from common import WrappedSequential


###############################################################################
# Test base fusion mechanism
###############################################################################

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
    if any(not m.fuseable for m in sequence):
        return None
    return sequence[0]


types_sequence = (DummyA, (DummyB, DummyC), DummyD)


def fused_reference():
    return WrappedSequential(DummyA(), nn.Identity(), nn.Identity())


def compare_models(actual, expected):
    nm_actual = OrderedDict(actual.named_modules())
    nm_expected = OrderedDict(expected.named_modules())
    assert nm_actual.keys() == nm_expected.keys()
    assert [type(v) for v in nm_actual.values()] == [type(v) for v in nm_expected.values()]


def fuse_and_check(model, expected, input_shape, parallel):
    if parallel:
        model = nn.DataParallel(model)
        expected = nn.DataParallel(expected)
    dummy_input = torch.randn(input_shape)
    fused = mt.fuse_modules(model, types_sequence=types_sequence, fuse_fn=fuse_fn, dummy_input=dummy_input)
    compare_models(fused, expected)


@pytest.fixture(params=[False, True], ids=['parallel_off', 'parallel_on'])
def parallel(request):
    return request.param


def test_fuse_modules(parallel):
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

    model = WrappedSequential(DummyB(), DummyD())
    fuse_and_check(model, deepcopy(model), input_shape, parallel)

    model = WrappedSequential(DummyA(), DummyB(), DummyA(), DummyD())
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


def test_fuse_modules_with_pre_exist_adj_map():
    model = WrappedSequential(DummyA(), DummyB(), DummyD())
    with pytest.raises(ValueError):
        mt.fuse_modules(model, types_sequence, fuse_fn, dummy_input=None, adjacency_map=None)

    dummy_input = torch.randn(10, 10)
    sg = SummaryGraph(deepcopy(model), dummy_input)
    adj_map = sg.adjacency_map()

    fused_dummy_input = mt.fuse_modules(deepcopy(model), types_sequence, fuse_fn,
                                        dummy_input=dummy_input, adjacency_map=None)
    compare_models(fused_dummy_input, fused_reference())

    fused_pre_sg = mt.fuse_modules(deepcopy(model), types_sequence, fuse_fn,
                                   dummy_input=None, adjacency_map=adj_map)
    compare_models(fused_pre_sg, fused_reference())

    fused_both = mt.fuse_modules(deepcopy(model), types_sequence, fuse_fn,
                                 dummy_input=dummy_input, adjacency_map=adj_map)
    compare_models(fused_both, fused_reference())


###############################################################################
# Test BN folding for inference
###############################################################################

# This warning seems to be a bug in batch_norm implementation, which compares a tensor to the value 1
@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect')
@pytest.mark.parametrize(
    'model, input_shape',
    [
        (WrappedSequential(nn.ReLU(), nn.BatchNorm1d(5)), (10, 5)),
        (WrappedSequential(nn.Conv1d(10, 20, 3), nn.ReLU()), (10, 10, 10)),
        (WrappedSequential(nn.Conv2d(10, 20, 3), nn.BatchNorm2d(20, track_running_stats=False)), (10, 10, 50, 50)),
        (WrappedSequential(nn.Linear(10, 20), nn.BatchNorm1d(20, track_running_stats=False)), (10, 10)),
        (BypassModel((nn.Conv2d(10, 20, 3),), nn.BatchNorm2d(20)), (10, 10, 50, 50))
    ],
    ids=['relu->bn', 'conv->relu', 'conv->bn_no_stats', 'linear->bn_no_stats', 'conv_multiple_outputs->bn']
)
def test_fold_batch_norms_inference_no_fold(model, input_shape):
    orig_model = deepcopy(model)
    folded_model = mt.fold_batch_norms(model, dummy_input=torch.randn(input_shape), inference=True)
    for (n_orig, m_orig), (n_folded, m_folded) in zip(orig_model.named_modules(), folded_model.named_modules()):
        assert n_folded == n_orig
        assert type(m_folded) == type(m_orig)

    for (n_orig, p_orig), (n_folded, p_folded) in zip(orig_model.named_parameters(), folded_model.named_parameters()):
        assert n_folded == n_orig
        assert (p_folded == p_orig).all().item() == 1


@pytest.mark.parametrize(
    'model, input_shape',
    [
        (WrappedSequential(nn.Conv1d(10, 20, 3), nn.BatchNorm1d(20)), (10, 10, 50)),
        (WrappedSequential(nn.Conv2d(10, 20, 3), nn.BatchNorm2d(20)), (10, 10, 50, 50)),
        (WrappedSequential(nn.Conv3d(10, 20, 3), nn.BatchNorm3d(20)), (10, 10, 20, 20, 20)),
        (WrappedSequential(nn.Linear(10, 20), nn.BatchNorm1d(20)), (10, 10))
    ],
    ids=['conv1d->bn1d', 'conv2d->bn2d', 'conv3d->bn3d', 'lienar->bn1d']
)
def test_fold_batch_norms_inference(model, input_shape):
    # Make sure we have non-trivial values to work with
    nn.init.uniform_(model.seq[1].weight)
    nn.init.uniform_(model.seq[1].bias)
    nn.init.uniform_(model.seq[1].running_mean)
    nn.init.uniform_(model.seq[1].running_var)

    model.eval()
    orig_model = deepcopy(model)
    dummy_input = torch.randn(input_shape)
    folded_model = mt.fold_batch_norms(model, dummy_input=dummy_input, inference=True)
    assert type(folded_model.seq[0]) == type(orig_model.seq[0])
    assert type(folded_model.seq[1]) == nn.Identity

    y_orig = orig_model(dummy_input)
    y_folded = folded_model(dummy_input)
    torch.testing.assert_allclose(y_folded, y_orig)
