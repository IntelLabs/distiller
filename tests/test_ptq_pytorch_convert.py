#
# Copyright (c) 2020 Intel Corporation
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

import torch
import torch.testing

import distiller.quantization as quantization
from distiller.quantization.range_linear import _get_quant_params_from_tensor


@pytest.fixture(params=[quantization.LinearQuantMode.SYMMETRIC,
                        quantization.LinearQuantMode.SYMMETRIC_RESTRICTED,
                        quantization.LinearQuantMode.ASYMMETRIC_UNSIGNED,
                        quantization.LinearQuantMode.ASYMMETRIC_SIGNED],
                ids=['symmetric', 'symmetric_restricted', 'asym_unsigned', 'asym_signed'])
def distiller_mode(request):
    return request.param


@pytest.fixture(params=[torch.qint8, torch.quint8], ids=['torch.qint8', 'torch.quint8'])
def torch_dtype(request):
    return request.param


@pytest.fixture(params=[False, True], ids=['per_tensor', 'per_channel'])
def per_channel(request):
    return request.param


@pytest.fixture(params=[False, True], ids=['reduce_off', 'reduce_on'])
def reduce_range(request):
    return request.param


@pytest.fixture()
def num_bits():
    return 8


@pytest.fixture()
def tensor():
    return torch.randn(64, 256, 7, 7)


def test_qparams_conversion(tensor, num_bits, distiller_mode, torch_dtype, per_channel, reduce_range):
    if reduce_range:
        if num_bits != 8:
            return True
        if quantization.is_linear_quant_mode_symmetric(distiller_mode) and torch_dtype == torch.quint8:
            return True

    # Calculate quantization parameters with Distiller for number of bits BEFORE reduce_range
    signed = distiller_mode != quantization.LinearQuantMode.ASYMMETRIC_UNSIGNED
    distiller_scale, distiller_zp = _get_quant_params_from_tensor(tensor, num_bits, distiller_mode,
                                                                  per_channel=per_channel)

    # Convert parameters to PyTorch
    converted_scale, converted_zp = quantization.distiller_qparams_to_pytorch(
        distiller_scale, distiller_zp, num_bits, distiller_mode, torch_dtype, reduce_range
    )

    # Quantize tensor with Distiller
    # If reduce_range is set, then we actually quantize with num_bits-1
    if reduce_range:
        num_bits -= 1
        distiller_scale, distiller_zp = _get_quant_params_from_tensor(tensor, num_bits, distiller_mode,
                                                                      per_channel=per_channel)
    restrict = distiller_mode == quantization.LinearQuantMode.SYMMETRIC_RESTRICTED
    clamp_min, clamp_max = quantization.get_quantized_range(num_bits, signed=signed, signed_restrict_qrange=restrict)
    distiller_q_t = quantization.linear_quantize_clamp(tensor, distiller_scale, distiller_zp, clamp_min, clamp_max)

    # Quantize with PyTorch
    if per_channel:
        pytorch_q_t = torch.quantize_per_channel(tensor, converted_scale, converted_zp, 0, torch_dtype)
    else:
        pytorch_q_t = torch.quantize_per_tensor(tensor, converted_scale, converted_zp, torch_dtype)

    # Dequantize
    distiller_q_dq_t = quantization.linear_dequantize(distiller_q_t, distiller_scale, distiller_zp)
    pytorch_q_dq_t = pytorch_q_t.dequantize()

    # Compare - allow of up to one quantized "bin" between the tensors
    if per_channel:
        for idx, scale in enumerate(converted_scale):
            torch.testing.assert_allclose(distiller_q_dq_t[idx], pytorch_q_dq_t[idx], atol=scale, rtol=1e-05)
    else:
        torch.testing.assert_allclose(pytorch_q_dq_t, distiller_q_dq_t, atol=converted_scale, rtol=1e-05)


def test_quantized_tensor_conversion(tensor, num_bits, distiller_mode, torch_dtype, per_channel):
    # Quantize tensor with Distiller
    signed = distiller_mode != quantization.LinearQuantMode.ASYMMETRIC_UNSIGNED
    distiller_scale, distiller_zp = _get_quant_params_from_tensor(tensor, num_bits, distiller_mode,
                                                                  per_channel=per_channel)
    restrict = distiller_mode == quantization.LinearQuantMode.SYMMETRIC_RESTRICTED
    clamp_min, clamp_max = quantization.get_quantized_range(num_bits, signed=signed, signed_restrict_qrange=restrict)
    distiller_q_t = quantization.linear_quantize_clamp(tensor, distiller_scale, distiller_zp, clamp_min, clamp_max)

    # Convert tensor to PyTorch
    pytorch_q_t = quantization.distiller_quantized_tensor_to_pytorch(
        distiller_q_t, distiller_scale, distiller_zp, num_bits, distiller_mode, torch_dtype, per_channel, 0
    )

    # Dequantize both
    distiller_q_dq_t = quantization.linear_dequantize(distiller_q_t, distiller_scale, distiller_zp)
    pytorch_q_dq_t = pytorch_q_t.dequantize()

    # Compare
    torch.testing.assert_allclose(pytorch_q_dq_t, distiller_q_dq_t)


#TODO: Add tests of full model conversion
