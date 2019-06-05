import distiller
import torch
from torch.testing import assert_allclose
import torch.nn as nn
from distiller.quantization.layer_fusions import FusedLinearBatchNorm
from copy import deepcopy
import pytest

ATOL = 5e-5
RTOL = 1e-3
BATCH_SIZE = 32


@pytest.fixture(name='has_bias', params=[False, True], ids=['bias_off', 'bias_on'])
def fixture_hasbias(request):
    return request.param


@pytest.fixture()
def linear_weight():
    t = torch.arange(1, 7, dtype=torch.float).view(3, 2)
    return nn.Parameter(t)


@pytest.fixture()
def linear_bias():
    t = torch.arange(7, 10, dtype=torch.float)
    return nn.Parameter(t)


@pytest.fixture()
def input_tensor():
    t = torch.arange(0, 4, dtype=torch.float).view(2, 2)
    return t


def test_folding_fc(has_bias, input_tensor, linear_weight, linear_bias):
    distiller.set_deterministic(1234)
    linear = nn.Linear(2, 3, bias=has_bias)
    linear.weight = linear_weight
    linear.bias = linear_bias if has_bias else None
    bn = nn.BatchNorm1d(3)
    folded = FusedLinearBatchNorm(deepcopy(linear), deepcopy(bn), quantized=False)
    x = input_tensor
    y_t_linear = linear(x)
    y_t = bn(y_t_linear)
    y_p = folded(x)
    assert_allclose(y_t, y_p, RTOL, ATOL)


