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
LR = 1e-3


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
    # test a couple more times:
    for _ in range(1, 10):
        x = torch.rand(batch_size, input_size)
        y_p = folded(x)
        y_t = bn(linear(x))
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
    conv2d, bn = conv2d.cuda(), bn.cuda()
    folded = folded.cuda()
    # test a couple more times:
    for _ in range(1, 10):
        x = torch.rand(batch_size, input_c, h, w, device="cuda")
        y_p = folded(x)
        y_t = bn(conv2d(x))
        assert_allclose(y_t, y_p, RTOL, ATOL)


@pytest.mark.parametrize(
    "batch_size, input_size, output_size",
    [
        (2, 2, 3),
        (32, 512, 1024),
        (256, 128, 1024)
    ]
)
def test_backprop_fc(has_bias, batch_size, input_size, output_size):
    distiller.set_deterministic(1234)
    linear = nn.Linear(input_size, output_size, bias=has_bias)
    bn = nn.BatchNorm1d(output_size)
    folded = FusedLinearBatchNorm(deepcopy(linear), deepcopy(bn), quantized=False)
    unfolded = nn.Sequential(linear, bn)
    folded, unfolded = folded.cuda(), unfolded.cuda()
    optimizer_folded = torch.optim.SGD(folded.parameters(), LR)
    optimizer_unfolded = torch.optim.SGD(unfolded.parameters(), LR)
    criterion = nn.MSELoss()
    for _ in range(10):
        # inputs and targets:
        x = torch.rand(batch_size, input_size, device="cuda")
        y_t = torch.rand_like(linear(x))
        # calc loss:
        optimizer_folded.zero_grad()
        optimizer_unfolded.zero_grad()
        loss_folded = criterion(folded(x), y_t)
        loss_unfolded = criterion(unfolded(x), y_t)
        # calc gradients:
        loss_folded.backward()
        loss_unfolded.backward()
        # check the gradients:
        assert_allclose(unfolded[0].weight.grad, folded.linear.weight.grad)
        if has_bias:
            # The bias of the linear layer doesn't participate in the calculation!
            # for more details - refer to `FusedLinearBatchNorm.forward`
            assert folded.linear.bias.grad is None
        assert_allclose(unfolded[1].weight.grad, folded.bn.weight.grad)
        assert_allclose(unfolded[1].bias.grad, folded.bn.bias.grad)
        # make a step:
        optimizer_unfolded.step()
        optimizer_folded.step()
        # check updated weights (we skip the linear bias)
        assert_allclose(unfolded[0].weight, folded.linear.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].weight, folded.bn.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].bias, folded.bn.bias, RTOL, ATOL)
        assert_allclose(unfolded[1].running_mean, folded.bn.running_mean, RTOL, ATOL)
        assert_allclose(unfolded[1].running_var, folded.bn.running_var, RTOL, ATOL)
    # testing evaluation:
    folded.eval()
    unfolded.eval()
    x = torch.rand(batch_size, input_size, device="cuda")
    assert_allclose(unfolded(x), folded(x), RTOL, ATOL)
        
        
@pytest.mark.parametrize(
    "batch_size, input_c, output_c, h, w, kernel_size",
    [
        (2, 2, 3, 224, 224, 3),
        (32, 3, 64, 224, 224, 3),
        (256, 3, 64, 28, 28, 7),
    ]
)
def test_backprop_conv(has_bias, batch_size, input_c, output_c, h, w, kernel_size):
    distiller.set_deterministic(1234)
    conv2d = nn.Conv2d(input_c, output_c, kernel_size, bias=has_bias)
    bn = nn.BatchNorm2d(output_c)
    folded = FusedLinearBatchNorm(deepcopy(conv2d), deepcopy(bn), quantized=False)
    unfolded = nn.Sequential(conv2d, bn)
    folded, unfolded = folded.cuda(), unfolded.cuda()
    optimizer_folded = torch.optim.SGD(folded.parameters(), LR)
    optimizer_unfolded = torch.optim.SGD(unfolded.parameters(), LR)
    criterion = nn.MSELoss()
    for _ in range(10):
        # inputs and targets:
        x = torch.rand(batch_size, input_c, h, w, device="cuda")
        y_t = torch.rand_like(conv2d(x))
        # calc loss:
        optimizer_folded.zero_grad()
        optimizer_unfolded.zero_grad()
        loss_folded = criterion(folded(x), y_t)
        loss_unfolded = criterion(unfolded(x), y_t)
        # calc gradients:
        loss_folded.backward()
        loss_unfolded.backward()
        # check the gradients:
        assert_allclose(unfolded[0].weight.grad, folded.linear.weight.grad, RTOL, ATOL)
        if has_bias:
            # The bias of the linear layer doesn't participate in the calculation!
            # for more details - refer to `FusedLinearBatchNorm.forward`
            assert folded.linear.bias.grad is None
        assert_allclose(unfolded[1].weight.grad, folded.bn.weight.grad, RTOL, ATOL)
        assert_allclose(unfolded[1].bias.grad, folded.bn.bias.grad, RTOL, ATOL)
        # make a step:
        optimizer_unfolded.step()
        optimizer_folded.step()
        # check updated weights (we skip the linear bias)
        assert_allclose(unfolded[0].weight, folded.linear.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].weight, folded.bn.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].bias, folded.bn.bias, RTOL, ATOL)
    # testing evaluation:
    folded.eval()
    unfolded.eval()
    x = torch.rand(batch_size, input_c, h, w, device="cuda")
    assert_allclose(unfolded(x), folded(x), RTOL, ATOL)



