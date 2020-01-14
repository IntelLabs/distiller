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

import torch
import pytest
from distiller.quantization import q_utils as qu
from common import pytest_raises_wrapper


def test_symmetric_qparams():
    with pytest.raises(ValueError):
        # Negative scalar
        qu.symmetric_linear_quantization_params(8, -5.)

        # Negative element in tensor
        qu.symmetric_linear_quantization_params(8, torch.tensor([-5., 10.]))

    # Scalar positive
    scale, zp = qu.symmetric_linear_quantization_params(8, 4., False)
    assert not isinstance(scale, torch.Tensor)
    assert not isinstance(zp, torch.Tensor)
    assert scale == 31.875
    assert zp == 0

    scale, zp = qu.symmetric_linear_quantization_params(8, 4., True)
    assert not isinstance(scale, torch.Tensor)
    assert not isinstance(zp, torch.Tensor)
    assert scale == 31.75
    assert zp == 0

    # Scalar positive integer
    scale, zp = qu.symmetric_linear_quantization_params(8, 4)
    assert not isinstance(scale, torch.Tensor)
    assert not isinstance(zp, torch.Tensor)
    assert scale == 31.875
    assert zp == 0

    scale, zp = qu.symmetric_linear_quantization_params(8, 4, True)
    assert not isinstance(scale, torch.Tensor)
    assert not isinstance(zp, torch.Tensor)
    assert scale == 31.75
    assert zp == 0

    # Scalar zero
    scale, zp = qu.symmetric_linear_quantization_params(8, 0.)
    assert scale == 1
    assert zp == 0

    scale, zp = qu.symmetric_linear_quantization_params(8, 0., True)
    assert scale == 1
    assert zp == 0

    # Tensor positives
    sat = torch.tensor([4., 10.])
    scale, zp = qu.symmetric_linear_quantization_params(8, sat)
    assert torch.equal(scale, torch.tensor([31.875, 12.75]))
    assert torch.equal(zp, torch.zeros_like(sat))

    scale, zp = qu.symmetric_linear_quantization_params(8, sat, True)
    assert torch.equal(scale, torch.tensor([31.75, 12.7]))
    assert torch.equal(zp, torch.zeros_like(sat))

    # Tensor positives - integer saturation values
    sat = torch.tensor([4, 10])
    scale, zp = qu.symmetric_linear_quantization_params(8, sat)
    assert torch.equal(scale, torch.tensor([31.875, 12.75]))
    assert torch.equal(zp, torch.zeros_like(sat, dtype=torch.float32))

    scale, zp = qu.symmetric_linear_quantization_params(8, sat, True)
    assert torch.equal(scale, torch.tensor([31.75, 12.7]))
    assert torch.equal(zp, torch.zeros_like(sat, dtype=torch.float32))

    # Tensor with 0
    sat = torch.tensor([4., 0.])
    scale, zp = qu.symmetric_linear_quantization_params(8, sat)
    assert torch.equal(scale, torch.tensor([31.875, 1.]))
    assert torch.equal(zp, torch.zeros_like(sat))

    scale, zp = qu.symmetric_linear_quantization_params(8, sat, True)
    assert torch.equal(scale, torch.tensor([31.75, 1.]))
    assert torch.equal(zp, torch.zeros_like(sat))


def test_asymmetric_qparams():
    with pytest.raises(ValueError):
        # Test min > max
        # min scalar, max scalar
        qu.asymmetric_linear_quantization_params(8, 5., 4.)
        # min scalar, max tensor
        qu.asymmetric_linear_quantization_params(8, 5., torch.tensor([5., 3.]))
        # min tensor, max scalar
        qu.asymmetric_linear_quantization_params(8, torch.tensor([5., 3.]), 4.)
        # min tensor, max tensor
        qu.asymmetric_linear_quantization_params(8, torch.tensor([5., 3.]), torch.tensor([4., 7.]))

    # min scalar, max scalar

    # Min negative, max positive
    scale, zp = qu.asymmetric_linear_quantization_params(8, -2., 10., integral_zero_point=True, signed=False)
    assert not isinstance(scale, torch.Tensor)
    assert not isinstance(zp, torch.Tensor)
    assert scale == 21.25
    assert zp == -42

    scale, zp = qu.asymmetric_linear_quantization_params(8, -2., 10., integral_zero_point=True, signed=True)
    assert scale == 21.25
    assert zp == 86

    scale, zp = qu.asymmetric_linear_quantization_params(8, -2., 10., integral_zero_point=False, signed=False)
    assert scale == 21.25
    assert zp == -42.5

    scale, zp = qu.asymmetric_linear_quantization_params(8, -2., 10., integral_zero_point=False, signed=True)
    assert scale == 21.25
    assert zp == 85.5

    # Integer saturation values
    scale, zp = qu.asymmetric_linear_quantization_params(8, -2, 10, integral_zero_point=False, signed=True)
    assert scale == 21.25
    assert zp == 85.5

    # Both positive
    scale, zp = qu.asymmetric_linear_quantization_params(8, 5., 10.)
    assert scale == 25.5
    assert zp == 0

    # Both negative
    scale, zp = qu.asymmetric_linear_quantization_params(8, -10., -5.)
    assert scale == 25.5
    assert zp == -255

    # Both zero
    scale, zp = qu.asymmetric_linear_quantization_params(8, 0., 0.)
    assert scale == 1.
    assert zp == 0

    # min scalar, max tensor
    scale, zp = qu.asymmetric_linear_quantization_params(8, -10., torch.tensor([-2., 5.]))
    assert torch.equal(scale,torch.tensor([25.5, 17]))
    assert torch.equal(zp, torch.tensor([-255., -170]))

    scale, zp = qu.asymmetric_linear_quantization_params(8, 0., torch.tensor([0., 5.]))
    assert torch.equal(scale, torch.tensor([1., 51.]))
    assert torch.equal(zp, torch.tensor([0., 0.]))

    # Integer saturation values
    scale, zp = qu.asymmetric_linear_quantization_params(8, -10., torch.tensor([-2, 5]))
    assert torch.equal(scale, torch.tensor([25.5, 17]))
    assert torch.equal(zp, torch.tensor([-255., -170]))

    # min tensor, max scalar
    scale, zp = qu.asymmetric_linear_quantization_params(8, torch.tensor([-2., 5.]), 10.)
    assert torch.equal(scale, torch.tensor([21.25, 25.5]))
    assert torch.equal(zp, torch.tensor([-42., 0.]))

    scale, zp = qu.asymmetric_linear_quantization_params(8, torch.tensor([0., -5.]), 0.)
    assert torch.equal(scale, torch.tensor([1., 51.]))
    assert torch.equal(zp, torch.tensor([0., -255.]))

    # Integer saturation values
    scale, zp = qu.asymmetric_linear_quantization_params(8, torch.tensor([-2, 5]), 10.)
    assert torch.equal(scale, torch.tensor([21.25, 25.5]))
    assert torch.equal(zp, torch.tensor([-42., 0.]))

    # min tensor, max tensor
    scale, zp = qu.asymmetric_linear_quantization_params(8,
                                                         torch.tensor([-2., 5., -10., 0.]),
                                                         torch.tensor([10., 10., -5., 0.]))
    assert torch.equal(scale, torch.tensor([21.25, 25.5, 25.5, 1.]))
    assert torch.equal(zp, torch.tensor([-42., 0., -255., 0.]))


test_tensor = torch.tensor([-93, 33, -77, -42, -89, -55, 79, -19, -94,
                            69, -46, -88, 19, -43, -38, 30, -56, 87,
                            27, -86, 81, -60, -19, 59, 58, -47, -26,
                            12, -74, 83, -83, -28, 69, 63, -95, -87]).float()
test_tensor_4d = test_tensor.reshape(2, 2, 3, 3)
test_tensor_2d = test_tensor.reshape(6, 6)

too_large_dim_msg = "Expecting ValueError when passing too large dim"


def test_get_tensor_min_max():
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_min_max, test_tensor_2d, per_dim=2)
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_min_max, test_tensor_2d, per_dim=6)

    t_min, t_max = qu.get_tensor_min_max(test_tensor_4d)
    assert torch.equal(t_min, torch.tensor(-95.))
    assert torch.equal(t_max, torch.tensor(87.))

    t_min, t_max = qu.get_tensor_min_max(test_tensor_4d, per_dim=0)
    assert torch.equal(t_min, torch.tensor([-94., -95.]))
    assert torch.equal(t_max, torch.tensor([87., 83.]))

    t_min, t_max = qu.get_tensor_min_max(test_tensor_2d, per_dim=0)
    assert torch.equal(t_min, torch.tensor([-93., -94., -56., -86., -74., -95.]))
    assert torch.equal(t_max, torch.tensor([33., 79., 87., 81., 83., 69.]))


def test_get_tensor_avg_min_max():
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_avg_min_max, test_tensor_2d, across_dim=2)
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_avg_min_max, test_tensor_2d, across_dim=6)

    t_min, t_max = qu.get_tensor_avg_min_max(test_tensor_2d)
    assert torch.equal(t_min, torch.tensor(-95.))
    assert torch.equal(t_max, torch.tensor(87.))

    t_min, t_max = qu.get_tensor_avg_min_max(test_tensor_2d, across_dim=0)
    assert torch.equal(t_min, torch.tensor(-83.))
    assert torch.equal(t_max, torch.tensor(72.))

    t_min, t_max = qu.get_tensor_avg_min_max(test_tensor_4d, across_dim=0)
    assert torch.equal(t_min, torch.tensor(-94.5))
    assert torch.equal(t_max, torch.tensor(85.))


def test_get_tensor_max_abs():
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_max_abs, test_tensor_2d, per_dim=2)
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_max_abs, test_tensor_2d, per_dim=6)

    t_abs = qu.get_tensor_max_abs(test_tensor_4d)
    assert torch.equal(t_abs, torch.tensor(95.))

    t_abs = qu.get_tensor_max_abs(test_tensor_4d, per_dim=0)
    assert torch.equal(t_abs, torch.tensor([94., 95.]))

    t_abs = qu.get_tensor_max_abs(test_tensor_2d, per_dim=0)
    assert torch.equal(t_abs, torch.tensor([93., 94., 87., 86., 83., 95.]))


def test_get_tensor_avg_max_abs():
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_avg_max_abs, test_tensor_2d, across_dim=2)
    pytest_raises_wrapper(ValueError, too_large_dim_msg, qu.get_tensor_avg_max_abs, test_tensor_2d, across_dim=6)

    t_abs = qu.get_tensor_avg_max_abs(test_tensor_2d)
    assert torch.equal(t_abs, torch.tensor(95.))

    t_abs = qu.get_tensor_avg_max_abs(test_tensor_2d, across_dim=0)
    assert torch.equal(t_abs, torch.tensor(83.))

    t_abs = qu.get_tensor_avg_max_abs(test_tensor_4d, across_dim=0)
    assert torch.equal(t_abs, torch.tensor(94.5))


def test_get_tensor_mean_n_stds_min_max():
    pytest_raises_wrapper(ValueError, 'Expecting ValueError with n_stds = 0',
                          qu.get_tensor_mean_n_stds_min_max, test_tensor, n_stds=0)

    mean = torch.tensor(-16.)
    std = torch.tensor(62.87447738647461)

    t_min, t_max = qu.get_tensor_mean_n_stds_min_max(test_tensor)
    torch.testing.assert_allclose(t_min, mean - std)
    torch.testing.assert_allclose(t_max, mean + std)

    t_min, t_max = qu.get_tensor_mean_n_stds_min_max(test_tensor, n_stds=2)
    torch.testing.assert_allclose(t_min, torch.tensor(-95.))
    torch.testing.assert_allclose(t_max, torch.tensor(87.))


@pytest.mark.parametrize(
    "num_bits, signed, restrict, expected_q_min, expected_q_max",
    [
        (8, False, False, 0, 255),
        (8, False, True, 0, 255),
        (8, True, False, -128, 127),
        (8, True, True, -127, 127),
    ]
)
def test_get_quantized_range(num_bits, signed, restrict, expected_q_min, expected_q_max):
    q_min, q_max = qu.get_quantized_range(num_bits, signed=signed, signed_restrict_qrange=restrict)
    assert q_min == expected_q_min
    assert q_max == expected_q_max


# TODO - Implement testing for ACIQ clipping
# def test_aciq_clipping():
#     pass
