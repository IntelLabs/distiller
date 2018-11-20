#
# Copyright (c) 2018 Intel Corporation
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


def symmetric_linear_quantization_params(num_bits, saturation_val):
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    scale = n / saturation_val
    if isinstance(scale, torch.Tensor):
        zero_point = torch.tensor(0.0).to(saturation_val.device)
    else:
        zero_point = 0.0
    return scale, zero_point


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    n = 2 ** num_bits - 1
    scale = n / (saturation_max - saturation_min)
    zero_point = scale * saturation_min
    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point.round_()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def get_tensor_min_max(t):
    return t.min(), t.max()


def get_tensor_avg_min_max_across_batch(t):
    # Assume batch is at dim 0
    tv = t.view(t.size()[0], -1)
    avg_max = tv.max(dim=1)[0].mean()
    avg_min = tv.min(dim=1)[0].mean()
    return avg_min, avg_max


def get_tensor_max_abs(t):
    min_val, max_val = get_tensor_min_max(t)
    return max(abs(min_val), abs(max_val))


def get_tensor_avg_max_abs_across_batch(t):
    avg_min, avg_max = get_tensor_avg_min_max_across_batch(t)
    return max(abs(avg_min), abs(avg_max))


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1
