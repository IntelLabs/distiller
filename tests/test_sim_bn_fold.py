import distiller
import torch
from torch.testing import assert_allclose
import torch.nn as nn
from distiller.quantization.sim_bn_fold import SimulatedFoldedBatchNorm
from copy import deepcopy
import pytest

ATOL = 2e-4
RTOL = 1e-3
BATCH_SIZE = 32
LR = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.mark.parametrize(
    "m1, m2",
    [
        (nn.ReLU(), nn.BatchNorm1d(5)),
        (nn.Conv1d(1, 2, 3), nn.ReLU()),
        (nn.Conv1d(1, 2, 3), nn.BatchNorm2d(2)),
        (nn.Conv2d(1, 2, 3), nn.BatchNorm3d(2)),
        (nn.Conv3d(1, 2, 3), nn.BatchNorm2d(2)),
        (nn.Linear(3, 5), nn.BatchNorm2d(5))
    ]
)
def test_simulated_bn_fold_bad_sequences(m1, m2):
    with pytest.raises(TypeError):
        SimulatedFoldedBatchNorm(m1, m2)


@pytest.fixture(params=[False, True], ids=['bias_off', 'bias_on'])
def has_bias(request):
    return request.param


@pytest.fixture(params=[0.1, None], ids=['ema', 'cma'])
def momentum(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['affine_on', 'affine_off'])
def affine(request):
    return request.param


@pytest.mark.parametrize(
    "batch_size, input_size, output_size",
    [
        (2, 2, 3),
        (32, 512, 1024),
        (256, 128, 1024)
    ]
)
def test_simulated_bn_fold_fc(has_bias, batch_size, input_size, output_size, momentum, affine):
    distiller.set_deterministic(1234)
    linear = nn.Linear(input_size, output_size, bias=has_bias)
    bn = nn.BatchNorm1d(output_size, momentum=momentum, affine=affine)
    run_simulated_bn_fold_test(linear, bn, (batch_size, input_size), has_bias)


@pytest.mark.parametrize(
    "batch_size, input_c, output_c, l, kernel_size",
    [
        (50, 3, 100, 80, 10),
    ]
)
def test_simulated_bn_fold_conv1d(has_bias, batch_size, input_c, output_c, l, kernel_size, momentum, affine):
    distiller.set_deterministic(1234)
    conv1d = nn.Conv1d(input_c, output_c, kernel_size, bias=has_bias)
    bn = nn.BatchNorm1d(output_c, momentum=momentum, affine=affine)
    run_simulated_bn_fold_test(conv1d, bn, (batch_size, input_c, l), has_bias)


@pytest.mark.parametrize(
    "batch_size, input_c, output_c, h, w, kernel_size",
    [
        (2, 2, 3, 224, 224, 3),
        (32, 3, 64, 224, 224, 3),
        (256, 3, 64, 28, 28, 7),
    ]
)
def test_simulated_bn_fold_conv2d(has_bias, batch_size, input_c, output_c, h, w, kernel_size, momentum, affine):
    distiller.set_deterministic(1234)
    conv2d = nn.Conv2d(input_c, output_c, kernel_size, bias=has_bias)
    bn = nn.BatchNorm2d(output_c, momentum=momentum, affine=affine)
    run_simulated_bn_fold_test(conv2d, bn, (batch_size, input_c, h, w), has_bias)


@pytest.mark.parametrize(
    "batch_size, input_c, output_c, h, w, d, kernel_size",
    [
        (2, 2, 3, 64, 64, 9, 3),
    ]
)
def test_simulated_bn_fold_conv3d(has_bias, batch_size, input_c, output_c, h, w, d, kernel_size, momentum, affine):
    distiller.set_deterministic(1234)
    conv3d = nn.Conv3d(input_c, output_c, kernel_size, bias=has_bias)
    bn = nn.BatchNorm3d(output_c, momentum=momentum, affine=affine)
    run_simulated_bn_fold_test(conv3d, bn, (batch_size, input_c, h, w, d), has_bias)


def run_simulated_bn_fold_test(param_layer, bn_layer, x_size, has_bias):
    folded = SimulatedFoldedBatchNorm(deepcopy(param_layer), deepcopy(bn_layer), param_quantization_fn=None)
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
        assert_allclose(unfolded[0].weight.grad, folded.param_module.weight.grad, RTOL, ATOL)
        if has_bias:
            # The bias of the linear layer doesn't participate in the calculation!
            # for more details - refer to `FusedLinearBatchNorm.forward`
            assert folded.param_module.bias.grad is None
        if bn_layer.affine:
            assert_allclose(unfolded[1].weight.grad, folded.bn.weight.grad, RTOL, ATOL)
            assert_allclose(unfolded[1].bias.grad, folded.bn.bias.grad, RTOL, ATOL)

        # make a step:
        optimizer_unfolded.step()
        optimizer_folded.step()

        # check updated weights (we skip the linear bias)
        assert_allclose(unfolded[0].weight, folded.param_module.weight, RTOL, ATOL)
        if bn_layer.affine:
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
