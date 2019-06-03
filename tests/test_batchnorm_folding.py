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


@pytest.fixture(name='bias', params=[False, True], ids=['bias_off', 'bias_on'])
def fixture_bias(request):
    return request.param


def test_folding_fc(bias):
    distiller.set_deterministic(1234)
    linear = nn.Linear(2, 1, bias=bias)
    bn = nn.BatchNorm1d(1)
    folded = FusedLinearBatchNorm(deepcopy(linear), deepcopy(bn), quantized=False)
    x = torch.rand(2, 2)
    y_t = linear(x)
    y_t = bn(y_t)
    y_p = folded(x)
    assert_allclose(y_t, y_p, RTOL, ATOL)


