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

import torch.nn as nn
from enum import Enum

from .quantizer import Quantizer
from .q_utils import *

###
# Range-based linear quantization
###


class LinearQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3


def get_tensor_quantization_params(tensor, num_bits, mode, clip, saturation_val_rescale=1):
    if mode == LinearQuantMode.SYMMETRIC:
        sat_fn = get_tensor_avg_max_abs_across_batch if clip else get_tensor_max_abs
        return symmetric_linear_quantization_params(num_bits, sat_fn(tensor) / saturation_val_rescale)
    else:   # Asymmetric mode
        sat_fn = get_tensor_avg_min_max_across_batch if clip else get_tensor_min_max
        sat_min, sat_max = sat_fn(tensor)
        sat_min /= saturation_val_rescale
        sat_max /= saturation_val_rescale
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        return asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)


class RangeLinearQuantWrapper(nn.Module):
    """
    Base class for module which wraps an existing module with linear range-base quantization functionality

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed/unsigned)
        clip_acts (bool): If true, will clip activations instead of using absolute min/max. At the moment clipping is
            done by averaging over the max absolute values of samples within a batch. More methods might be added in
            the future.
    """

    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC,
                 clip_acts=False):
        super(RangeLinearQuantWrapper, self).__init__()

        self.wrapped_module = wrapped_module
        self.num_bits_acts = num_bits_acts
        self.num_bits_accum = num_bits_accum
        self.clip_acts = clip_acts
        self.mode = mode

        # Controls whether output is de-quantized at end of forward op. Meant as a debug / test flag only
        # (note that if False, the quantized output will be returned, but without any quantization parameters,
        #  so other than inspecting the contents there's not much to do with it)
        self._dequant_out = True

        signed = mode != LinearQuantMode.ASYMMETRIC_UNSIGNED
        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=signed)
        # The accumulator is always signed
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True)

    def forward(self, *inputs):
        if self.training:
            raise RuntimeError(self.__class__.__name__ + " can only be used in eval mode")

        in_scales, in_zero_points = self.get_inputs_quantization_params(*inputs)

        # Quantize inputs
        inputs_q = []
        for idx, input in enumerate(inputs):
            input_q = linear_quantize_clamp(input.data, in_scales[idx], in_zero_points[idx],
                                            self.acts_min_q_val, self.acts_max_q_val, inplace=False)
            inputs_q.append(torch.autograd.Variable(input_q))

        # Forward through wrapped module
        accum = self.quantized_forward(*inputs_q)

        # Re-quantize accumulator to quantized output range
        out_scale, out_zero_point = self.get_output_quantization_params(accum)
        requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point)
        out_q = linear_quantize_clamp(accum.data, requant_scale, requant_zero_point,
                                      self.acts_min_q_val, self.acts_max_q_val, inplace=True)

        if not self._dequant_out:
            return torch.autograd.Variable(out_q)

        # De-quantize back to FP32
        out_f = linear_dequantize(out_q, out_scale, out_zero_point, inplace=True)

        return torch.autograd.Variable(out_f)

    def get_inputs_quantization_params(self, *inputs):
        """
        Calculate input quantization parameters (scale and zero-point)

        Should be overridden by all subclasses

        :param inputs: Current input tensors passed to forward method
        :return: Tuple of 2 lists - list of scales per input and list of zero-point per input
        """
        raise NotImplementedError

    def quantized_forward(self, *inputs_q):
        """
        Perform forward pass with quantized inputs and return quantized outputs

        :param inputs_q: Tensor (or list of tensors) with quantized input values
        :return: Tensor with quantized output values
        """
        raise NotImplementedError

    def get_output_quantization_params(self, accumulator):
        """
        Calculate quantization parameters (scale and zero-point) for the output.
        This is used for:
          * Calculating the accumulator-to-output re-quantization parameters
            (see get_accum_to_output_re_quantization_params)
          * De-quantizing the output back to FP32

        Should be overridden by all subclasses

        :param accumulator: Tensor with accumulator values
        :return: Tuple of scale and zero-point
        """
        raise NotImplementedError

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        """
        Calculate quantization parameters (scale and zero-point) for re-quantization, that is:
        Converting the intermediate integer accumulator to the output range

        Should be overridden by all subclasses

        :param output_scale: Output scale factor
        :param output_zero_point: Output zero-point
        :return: Tuple of scale and zero-point
        """
        raise NotImplementedError


class RangeLinearQuantParamLayerWrapper(RangeLinearQuantWrapper):
    """
    Linear range-based quantization wrappers for layers with weights and bias (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Assume:

    x_q = round(scale_x * x_f) - zero_point_x

    Hence:

    x_f = 1/scale_x * x_q + zero_point_x

    (And the same for y_q, w_q and b_q)

    So, we get: (use "zp" as abbreviation for zero_point)

    y_f = x_f * w_f + b_f

    y_q = scale_y * y_f + zp_y =  scale_y * (x_f * w_f + b_f) + zp_y =

                scale_y                                         scale_x * scale_w
        = ------------------- * ((x_q + zp_x) * (w_q + zp_w) + ------------------- * (b_q + zp_b)) + zp_y
           scale_x * scale_w                                         scale_b

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights and bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed/unsigned)
        clip_acts (bool): See RangeLinearQuantWrapper
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 mode=LinearQuantMode.SYMMETRIC, clip_acts=False):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum, mode,
                                                                clip_acts)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D and Linear modules')

        self.num_bits_params = num_bits_params
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            num_bits_params, signed=mode != LinearQuantMode.ASYMMETRIC_UNSIGNED)

        # Quantize weights - overwrite FP32 weights
        w_scale, w_zero_point = get_tensor_quantization_params(wrapped_module.weight, num_bits_params, self.mode, False)
        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_zero_point', w_zero_point)
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point, self.params_min_q_val,
                              self.params_max_q_val, inplace=True)

        # Quantize bias
        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        if self.has_bias:
            b_scale, b_zero_point = get_tensor_quantization_params(wrapped_module.bias, num_bits_params, self.mode,
                                                                   False)
            self.register_buffer('b_scale', b_scale)
            self.register_buffer('b_zero_point', b_zero_point)
            base_b_q = linear_quantize_clamp(wrapped_module.bias.data, self.b_scale, self.b_zero_point,
                                             self.params_min_q_val, self.params_max_q_val)
            # Dynamic ranges - save in auxiliary buffer, requantize each time based on dynamic input scale factor
            self.register_buffer('base_b_q', base_b_q)

        self.current_in_scale = 1
        self.current_in_zero_point = 0
        self.current_accum_scale = 1

    def get_inputs_quantization_params(self, input):
        self.current_in_scale, self.current_in_zero_point = get_tensor_quantization_params(input, self.num_bits_acts,
                                                                                           self.mode, self.clip_acts)
        return [self.current_in_scale], [self.current_in_zero_point]

    def quantized_forward(self, input_q):
        # See class documentation for quantized calculation details.
        # Note the main terms within the summation is:
        #   (x_q + zp_x) * (w_q + zp_w)
        # In a performance-optimized solution, we would expand the parentheses and perform the computation similar
        # to what is described here:
        #   https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
        # However, for now we're more concerned with simplicity rather than speed. So we'll just add the zero points
        # to the input and weights and pass those to the wrapped model. Functionally, since at this point we're
        # dealing solely with integer values, the results are the same either way.

        self.current_accum_scale = self.current_in_scale * self.w_scale
        if self.has_bias:
            # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
            self.wrapped_module.bias.data = linear_quantize_clamp(self.base_b_q,
                                                                  self.current_accum_scale / self.b_scale,
                                                                  -self.b_zero_point,
                                                                  self.accum_min_q_val, self.accum_max_q_val)

        if self.mode != LinearQuantMode.SYMMETRIC:
            input_q += self.current_in_zero_point
            self.wrapped_module.weight.data += self.w_zero_point

        accum = self.wrapped_module.forward(input_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        if self.mode != LinearQuantMode.SYMMETRIC:
            self.wrapped_module.weight.data -= self.w_zero_point
        return accum

    def get_output_quantization_params(self, accumulator):
        return get_tensor_quantization_params(accumulator, self.num_bits_acts, self.mode, self.clip_acts,
                                              self.current_accum_scale)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return output_scale / self.current_accum_scale, output_zero_point

    def extra_repr(self):
        tmpstr = 'mode={0}, '.format(str(self.mode).split('.')[1])
        tmpstr += 'num_bits_acts={0}, num_bits_params={1}, num_bits_accum={2}, '.format(self.num_bits_acts,
                                                                                        self.num_bits_params,
                                                                                        self.num_bits_accum)
        tmpstr += 'clip_acts={0}'.format(self.clip_acts)
        return tmpstr


class PostTrainLinearQuantizer(Quantizer):
    """
    Applies range-based linear quantization to a model.
    This quantizer is expected to be executed at evaluation only, on a pre-trained model
    Currently, the following Modules are supported: torch.nn.Conv2d, torch.nn.Linear

    Args:
        model (torch.nn.Module): Model to be quantized
        bits_activations/parameters/accum (int): Number of bits to be used when quantizing each tensor type
        clip_acts (bool): See RangeLinearQuantWrapper
        no_clip_layers (list): List of fully-qualified layer names for which activations clipping should not be done.
            A common practice is to not clip the activations of the last layer before softmax.
            Applicable only if clip_acts is True.
    """
    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32, mode=LinearQuantMode.SYMMETRIC,
                 clip_acts=False, no_clip_layers=[]):
        super(PostTrainLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters, train_with_fp_copy=False)

        if isinstance(mode, str):
            try:
                mode = LinearQuantMode[mode]
            except KeyError:
                raise ValueError('Unknown quantization mode string')
        elif isinstance(mode, LinearQuantMode):
            mode = mode
        else:
            raise TypeError("'mode' argument can be either a string or member of {0}".format(LinearQuantMode.__name__))
        
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters,
                                                    'bits_accum': bits_accum,
                                                    'mode': str(mode).split('.')[1], 'clip_acts': clip_acts,
                                                    'no_clip_layers': no_clip_layers}}
        
        def replace_fn(module, name, qbits_map):
            clip = self.clip_acts and name not in no_clip_layers
            return RangeLinearQuantParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                                     num_bits_accum=self.bits_accum, mode=mode, clip_acts=clip)

        self.clip_acts = clip_acts
        self.no_clip_layers = no_clip_layers
        self.bits_accum = bits_accum
        self.mode = mode
        self.replacement_factory[nn.Conv2d] = replace_fn
        self.replacement_factory[nn.Linear] = replace_fn
