import distiller
import torch
from torch.testing import assert_allclose
import torch.nn as nn
from distiller.quantization.layer_fusions import SimulatedFoldedBatchNorm
from copy import deepcopy
import pytest

ATOL = 5e-5
RTOL = 1e-3
BATCH_SIZE = 32
LR = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(params=[False, True], ids=['bias_off', 'bias_on'])
def has_bias(request):
    return request.param


@pytest.fixture(params=[0.1, None], ids=['ema', 'cma'])
def momentum(request):
    return request.param


@pytest.mark.parametrize(
    "batch_size, input_size, output_size",
    [
        (2, 2, 3),
        (32, 512, 1024),
        (256, 128, 1024)
    ]
)
def test_simulated_bn_fold_fc(has_bias, batch_size, input_size, output_size, momentum):
    distiller.set_deterministic(1234)
    linear = nn.Linear(input_size, output_size, bias=has_bias)
    bn = nn.BatchNorm1d(output_size, momentum=momentum)
    run_simulated_bn_fold_test(linear, bn, (batch_size, input_size), has_bias)
        

@pytest.mark.parametrize(
    "batch_size, input_c, output_c, h, w, kernel_size",
    [
        (2, 2, 3, 224, 224, 3),
        (32, 3, 64, 224, 224, 3),
        (256, 3, 64, 28, 28, 7),
    ]
)
def test_simulated_bn_fold_conv(has_bias, batch_size, input_c, output_c, h, w, kernel_size, momentum):
    distiller.set_deterministic(1234)
    conv2d = nn.Conv2d(input_c, output_c, kernel_size, bias=has_bias)
    bn = nn.BatchNorm2d(output_c, momentum=momentum)
    run_simulated_bn_fold_test(conv2d, bn, (batch_size, input_c, h, w), has_bias)


def run_simulated_bn_fold_test(param_layer, bn_layer, x_size, has_bias):
    folded = SimulatedFoldedBatchNorm(deepcopy(param_layer), deepcopy(bn_layer), quantized=False)
    unfolded = nn.Sequential(param_layer, bn_layer)
    folded, unfolded = folded.to(DEVICE), unfolded.to(DEVICE)
    optimizer_folded = torch.optim.SGD(folded.parameters(), LR)
    optimizer_unfolded = torch.optim.SGD(unfolded.parameters(), LR)
    criterion = nn.MSELoss().to(DEVICE)

    # Test for 10 "epochs" (train + eval)
    for _ in range(10):
        folded.train()
        unfolded.train()

        # inputs and targets:
        x = torch.rand(x_size, device=DEVICE)
        y_t = torch.rand_like(param_layer(x))

        # calc loss:
        optimizer_folded.zero_grad()
        optimizer_unfolded.zero_grad()
        loss_folded = criterion(folded(x), y_t)
        loss_unfolded = criterion(unfolded(x), y_t)

        # calc gradients:
        loss_folded.backward()
        loss_unfolded.backward()

        # check the gradients:
        assert_allclose(unfolded[0].weight.grad, folded.param_module.weight.grad)
        if has_bias:
            # The bias of the linear layer doesn't participate in the calculation!
            # for more details - refer to `FusedLinearBatchNorm.forward`
            assert folded.param_module.bias.grad is None
        assert_allclose(unfolded[1].weight.grad, folded.bn.weight.grad)
        assert_allclose(unfolded[1].bias.grad, folded.bn.bias.grad)

        # make a step:
        optimizer_unfolded.step()
        optimizer_folded.step()

        # check updated weights (we skip the linear bias)
        assert_allclose(unfolded[0].weight, folded.param_module.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].weight, folded.bn.weight, RTOL, ATOL)
        assert_allclose(unfolded[1].bias, folded.bn.bias, RTOL, ATOL)
        assert_allclose(unfolded[1].running_mean, folded.bn.running_mean, RTOL, ATOL)
        assert_allclose(unfolded[1].running_var, folded.bn.running_var, RTOL, ATOL)

        # testing evaluation:
        folded.eval()
        unfolded.eval()
        x = torch.rand(x_size, device=DEVICE)
        assert_allclose(unfolded(x), folded(x), RTOL, ATOL)

    # test eval after freezing
    folded.freeze()
    x = torch.rand(x_size, device=DEVICE)
    assert_allclose(unfolded(x), folded(x), RTOL, ATOL)
