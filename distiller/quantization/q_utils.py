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

from enum import Enum
import torch


def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def symmetric_linear_quantization_params(num_bits, saturation_val):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
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


def get_tensor_min(t, per_dim=None):
    if per_dim is None:
        return t.min()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0]


def get_tensor_max(t, per_dim=None):
    if per_dim is None:
        return t.max()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.max(dim=-1)[0]


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def get_tensor_avg_min_max(t, across_dim=None):
    min_per_dim, max_per_dim = get_tensor_min_max(t, per_dim=across_dim)
    return min_per_dim.mean(), max_per_dim.mean()


def get_tensor_max_abs(t, per_dim=None):
    min_val, max_val = get_tensor_min_max(t, per_dim=per_dim)
    return torch.max(min_val.abs_(), max_val.abs_())


def get_tensor_avg_max_abs(t, across_dim=None):
    avg_min, avg_max = get_tensor_avg_min_max(t, across_dim=across_dim)
    return torch.max(avg_min.abs_(), avg_max.abs_())


class AciqClipper:
    class AciqClippingType(Enum):
        Laplace = 1
        Gauss = 2

    def get_alpha_laplace(self, t, across_dim=None, num_bits=8, half_range=False):
        alpha_laplace = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        alpha_laplace_positive = {2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}

        # Mean of means across dims is equavalent to gloabal mean
        b = torch.mean(torch.abs(t - t.mean()))
        return b * (alpha_laplace_positive[num_bits] if half_range else alpha_laplace[num_bits])

    def get_alpha_gauss(self, t, across_dim=None, num_bits=8, half_range=False):
        alpha_gaus = {2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        alpha_gaus_positive = {2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92,
                               8: 4.1}  # TODO: add 8 bit multiplier

        # Mean of means across dims is equavalent to gloabal mean
        std = torch.std(t)
        return std * (alpha_gaus_positive[num_bits] if half_range else alpha_gaus[num_bits])


class AciqSymetricClipper(AciqClipper):
    def __init__(self, num_bits, clip_type=AciqClipper.AciqClippingType.Laplace):
        self.num_bits = num_bits
        self.clip_type = clip_type

    def __call__(self, t, across_dim=None):
        if self.clip_type == AciqClipper.AciqClippingType.Laplace:
            alpha = super(AciqSymetricClipper, self).get_alpha_laplace(t, across_dim, self.num_bits)
        else:
            alpha = super(AciqSymetricClipper, self).get_alpha_gauss(t, across_dim, self.num_bits)
        mean = t.mean()
        return torch.abs(mean) + alpha


class AciqAsymetricClipper(AciqClipper):
    def __init__(self, num_bits, clip_type=AciqClipper.AciqClippingType.Laplace, half_range=False):
        self.num_bits = num_bits
        self.clip_type = clip_type
        self.half_range = half_range

    def __call__(self, t, across_dim=None):
        if self.clip_type == AciqClipper.AciqClippingType.Laplace:
            alpha = super(AciqAsymetricClipper, self).get_alpha_laplace(t, across_dim, self.num_bits, half_range=self.half_range)
        else:
            alpha = super(AciqAsymetricClipper, self).get_alpha_gauss(t, across_dim, self.num_bits, half_range=self.half_range)

        mean = t.mean()
        min_val = get_tensor_min(t, across_dim).mean()
        min_val = torch.max(min_val, mean - alpha)

        return min_val, min_val + 2 * alpha


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace, clamp_minq=None, clamp_maxq=None):
        if inplace:
            ctx.mark_dirty(input)
        if clamp_minq is not None and clamp_maxq is not None:
            output = linear_quantize_clamp(input, scale, zero_point, clamp_minq, clamp_maxq, inplace)
        else:
            output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None, None, None
