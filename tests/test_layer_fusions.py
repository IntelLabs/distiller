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


@pytest.mark.parametrize(
    "batch_size, input_size, output_size",
    [
        (2, 2, 3),
        (32, 512, 1024),
        (256, 128, 1024)
    ]
)
def test_folding_fc(has_bias, batch_size, input_size, output_size):
    distiller.set_deterministic(1234)
    linear = nn.Linear(input_size, output_size, bias=has_bias)
    x = torch.rand(batch_size, input_size)
    bn = nn.BatchNorm1d(output_size)
    folded = FusedLinearBatchNorm(deepcopy(linear), deepcopy(bn), quantized=False)
    y_t_linear = linear(x)
    y_t = bn(y_t_linear)
    y_p = folded(x)
    assert_allclose(y_t, y_p, RTOL, ATOL)


@pytest.mark.parametrize(
    "batch_size, input_c, output_c, h, w, kernel_size",
    [
        (2, 2, 3, 224, 224, 3),
        (32, 3, 64, 224, 224, 3),
        (256, 3, 64, 28, 28, 7),
    ]
)
def test_folding_conv(has_bias, batch_size, input_c, output_c, h, w, kernel_size):
    distiller.set_deterministic(1234)
    conv2d = nn.Conv2d(input_c, output_c, kernel_size, bias=has_bias)
    x = torch.rand(batch_size, input_c, h, w)
    bn = nn.BatchNorm2d(output_c)
    folded = FusedLinearBatchNorm(deepcopy(conv2d), deepcopy(bn), quantized=False)
    y_t_linear = conv2d(x)
    y_t = bn(y_t_linear)
    y_p = folded(x)
    assert_allclose(y_t, y_p, RTOL, ATOL)



