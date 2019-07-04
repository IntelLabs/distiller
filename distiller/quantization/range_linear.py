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
import argparse
from enum import Enum
from collections import OrderedDict
from functools import reduce, partial
import logging
import os

import distiller
import distiller.utils
from .quantizer import Quantizer
from .q_utils import *
import distiller.modules

msglogger = logging.getLogger()


def _enum_to_str(enum_val):
    return str(enum_val).split('.')[1]


class LinearQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3


class ClipMode(Enum):
    # No clipping - absolute min/max values will be used
    NONE = 0
    # Clip value calculated by averaging over the max absolute values of samples within a batch
    AVG = 1
    # Clip value calculated as mean of tensor + N standard deviations. N should be specified separately
    N_STD = 2


def _verify_enum_value(val, enum_cls):
    cls_name = enum_cls.__name__
    if isinstance(val, str):
        try:
            return enum_cls[val]
        except KeyError:
            raise ValueError("Input string '{0}' doesn't match any of the values of {1}: {2}"
                             .format(val, cls_name, [e.name for e in enum_cls]))
    elif isinstance(val, enum_cls):
        return val
    else:
        raise TypeError("Argument can be either a string or member of {0} (got {1})".format(cls_name, val))


def verify_quant_mode(mode):
    return _verify_enum_value(mode, LinearQuantMode)


def verify_clip_mode(mode):
    return _verify_enum_value(mode, ClipMode)


def _get_saturation_fn(quant_mode, clip_mode, num_stds):
    if quant_mode == LinearQuantMode.SYMMETRIC:
        fns = {ClipMode.NONE: get_tensor_max_abs,
               ClipMode.AVG: get_tensor_avg_max_abs,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_max_abs, n_stds=num_stds)}
    else:  # Asymmetric mode
        fns = {ClipMode.NONE: get_tensor_min_max,
               ClipMode.AVG: get_tensor_avg_min_max,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_min_max, n_stds=num_stds)}
    return fns[clip_mode]


def _get_quant_params_from_tensor(tensor, num_bits, mode, clip=ClipMode.NONE, per_channel=False, num_stds=None,
                                  scale_approx_mult_bits=None):
    if per_channel and tensor.dim() not in [2, 4]:
        raise ValueError('Per channel quantization possible only with 2D or 4D tensors (linear or conv layer weights)')

    if clip == ClipMode.N_STD:
        if per_channel:
            raise ValueError('N_STD clipping not supported with per-channel quantization')
        if num_stds is None:
            raise ValueError('Clip mode set top N_STD but \'num_stds\' parameter not provided')

    dim = 0 if clip == ClipMode.AVG or per_channel else None
    sat_fn = _get_saturation_fn(mode, clip, num_stds)
    if mode == LinearQuantMode.SYMMETRIC:
        sat_val = sat_fn(tensor, dim)
        scale, zp = symmetric_linear_quantization_params(num_bits, sat_val)
    else:   # Asymmetric mode
        sat_min, sat_max = sat_fn(tensor, dim)
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if per_channel:
        # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
        dims = [scale.shape[0]] + [1] * (tensor.dim() - 1)
        scale = scale.view(dims)
        zp = zp.view(dims)

    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)

    return scale, zp


def _get_quant_params_from_stats_dict(stats, num_bits, mode, clip=ClipMode.NONE, num_stds=None,
                                      scale_approx_mult_bits=None):
    if clip == ClipMode.N_STD:
        if num_stds is None:
            raise ValueError('Clip mode set to N_STD but \'num_stds\' parameter not provided')
        if num_stds <= 0:
            raise ValueError('n_stds must be > 0, got {}'.format(num_stds))
    prefix = 'avg_' if clip == ClipMode.AVG else ''
    sat_min = torch.tensor(float(stats[prefix + 'min']))
    sat_max = torch.tensor(float(stats[prefix + 'max']))
    if clip == ClipMode.N_STD:
        mean = torch.tensor(float(stats['mean']))
        std = torch.tensor(float(stats['std']))
        sat_min = torch.max(sat_min, mean - num_stds * std)
        sat_max = torch.min(sat_max, mean + num_stds * std)
    if mode == LinearQuantMode.SYMMETRIC:
        scale, zp = symmetric_linear_quantization_params(num_bits, torch.max(sat_min.abs_(), sat_max.abs_()))
    else:
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)

    return scale, zp


###############################################################################
# Post Training
###############################################################################


def add_post_train_quant_args(argparser):
    str_to_quant_mode_map = {'sym': LinearQuantMode.SYMMETRIC,
                             'asym_s': LinearQuantMode.ASYMMETRIC_SIGNED,
                             'asym_u': LinearQuantMode.ASYMMETRIC_UNSIGNED}

    str_to_clip_mode_map = {'none': ClipMode.NONE, 'avg': ClipMode.AVG, 'n_std': ClipMode.N_STD}

    def from_dict(d, val_str):
        try:
            return d[val_str]
        except KeyError:
            raise argparse.ArgumentTypeError('Must be one of {0} (received {1})'.format(list(d.keys()), val_str))

    linear_quant_mode_str = partial(from_dict, str_to_quant_mode_map)
    clip_mode_str = partial(from_dict, str_to_clip_mode_map)

    group = argparser.add_argument_group('Arguments controlling quantization at evaluation time '
                                         '("post-training quantization")')
    exc_group = group.add_mutually_exclusive_group()
    exc_group.add_argument('--quantize-eval', '--qe', action='store_true',
                       help='Apply linear quantization to model before evaluation. Applicable only if '
                            '--evaluate is also set')
    exc_group.add_argument('--qe-calibration', type=distiller.utils.float_range_argparse_checker(exc_min=True),
                           metavar='PORTION_OF_TEST_SET',
                           help='Run the model in evaluation mode on the specified portion of the test dataset and '
                                'collect statistics. Ignores all other \'qe--*\' arguments')
    group.add_argument('--qe-mode', '--qem', type=linear_quant_mode_str, default='sym',
                       help='Linear quantization mode. Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of activations')
    group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of weights')
    group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                       help='Number of bits for quantization of the accumulator')
    group.add_argument('--qe-clip-acts', '--qeca', type=clip_mode_str, default='none',
                       help='Activations clipping mode. Choices: ' + ' | '.join(str_to_clip_mode_map.keys()))
    group.add_argument('--qe-clip-n-stds', type=float,
                       help='When qe-clip-acts is set to \'n_std\', this is the number of standard deviations to use')
    group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                       help='List of layer names for which not to clip activations. Applicable '
                            'only if --qe-clip-acts is not \'none\'')
    group.add_argument('--qe-per-channel', '--qepc', action='store_true',
                       help='Enable per-channel quantization of weights (per output channel)')
    group.add_argument('--qe-scale-approx-bits', '--qesab', type=int, metavar='NUM_BITS',
                       help='Enables scale factor approximation using integer multiply + bit shift, using '
                            'this number of bits the integer multiplier')
    group.add_argument('--qe-stats-file', type=str, metavar='PATH',
                       help='Path to YAML file with calibration stats. If not given, dynamic quantization will '
                            'be run (Note that not all layer types are supported for dynamic quantization)')
    group.add_argument('--qe-config-file', type=str, metavar='PATH',
                       help='Path to YAML file containing configuration for PostTrainLinearQuantizer (if present, '
                            'all other --qe* arguments are ignored)')


class RangeLinearQuantWrapper(nn.Module):
    """
    Base class for module which wraps an existing module with linear range-base quantization functionality

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed / unsigned)
        clip_acts (ClipMode): Activations clipping mode to use
        activation_stats (dict): Dict containing activation stats, used for static calculation of quantization
            parameters. Dict should be in the format exported by distiller.data_loggers.QuantCalibrationStatsCollector.
            If None then parameters are calculated dynamically.
        clip_n_stds (float): When clip_acts == ClipMode.N_STD, this is the number of standard deviations to use
        scale_approx_mult_bits (int): If not None, scale factors will be approximated using an integer multiplication
            followed by a bit-wise shift. This eliminates floating-point scale factors, replacing them with integer
            calculations.
            If None, scale factors will be kept in their original FP32 values.
    """

    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC,
                 clip_acts=ClipMode.NONE, activation_stats=None, clip_n_stds=None, scale_approx_mult_bits=None):
        super(RangeLinearQuantWrapper, self).__init__()

        self.wrapped_module = wrapped_module
        self.num_bits_acts = num_bits_acts
        self.num_bits_accum = num_bits_accum
        self.mode = mode
        self.clip_acts = clip_acts
        self.clip_n_stds = clip_n_stds
        self.scale_approx_mult_bits = scale_approx_mult_bits

        # Controls whether output is de-quantized at end of forward op. Meant as a debug / test flag only
        # (note that if False, the quantized output will be returned, but without any quantization parameters,
        #  so other than inspecting the contents there's not much to do with it)
        self._dequant_out = True

        signed = mode != LinearQuantMode.ASYMMETRIC_UNSIGNED
        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=signed)
        # The accumulator is always signed
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True)

        if activation_stats:
            self.preset_act_stats = True
            self.num_inputs = 0
            for idx, stats in activation_stats['inputs'].items():
                self.num_inputs += 1
                scale, zp = _get_quant_params_from_stats_dict(stats, num_bits_acts, mode, clip_acts, clip_n_stds,
                                                              scale_approx_mult_bits)
                prefix = 'in_{0}_'.format(idx)
                self.register_buffer(prefix + 'scale', scale)
                self.register_buffer(prefix + 'zero_point', zp)

            scale, zp = _get_quant_params_from_stats_dict(activation_stats['output'], num_bits_acts, mode, clip_acts,
                                                          clip_n_stds, scale_approx_mult_bits)
            self.register_buffer('output_scale', scale)
            self.register_buffer('output_zero_point', zp)
        else:
            self.preset_act_stats = False

    def inputs_scales(self):
        for scale in self._inputs_qparam('scale'):
            yield scale

    def inputs_zero_points(self):
        for zp in self._inputs_qparam('zero_point'):
            yield zp

    def _inputs_qparam(self, type_str):
        if type_str not in ['scale', 'zero_point']:
            raise ValueError('Unknown quantization parameter type')
        if not self.preset_act_stats:
            raise RuntimeError('Input quantization parameter iterators only available when activation stats were given')
        for idx in range(self.num_inputs):
            name = 'in_{0}_{1}'.format(idx, type_str)
            yield getattr(self, name)

    def forward(self, *inputs):
        if self.training:
            raise RuntimeError(self.__class__.__name__ + " can only be used in eval mode")
        device = inputs[0].device
        for buffer_name, buffer in self._buffers.items():
            setattr(self, buffer_name, buffer.to(device))

        in_scales, in_zero_points = self.get_inputs_quantization_params(*inputs)

        # Quantize inputs
        inputs_q = [linear_quantize_clamp(input.data, scale, zp,
                                          self.acts_min_q_val, self.acts_max_q_val, inplace=False)
                    for input, scale, zp in zip(inputs, in_scales, in_zero_points)]

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

        return out_f

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

    def extra_repr(self):
        tmpstr = 'mode={0}, '.format(str(self.mode).split('.')[1])
        tmpstr += 'num_bits_acts={0}, num_bits_accum={1}, '.format(self.num_bits_acts, self.num_bits_accum)
        tmpstr += 'clip_acts={0}, '.format(_enum_to_str(self.clip_acts))
        if self.clip_acts == ClipMode.N_STD:
            tmpstr += 'num_stds={} '.format(self.clip_n_stds)
        tmpstr += 'scale_approx_mult_bits={}'.format(self.scale_approx_mult_bits)
        tmpstr += '\npreset_activation_stats={0}'.format(self.preset_act_stats)
        if self.preset_act_stats:
            for idx, (in_scale, in_zp) in enumerate(zip(self.inputs_scales(), self.inputs_zero_points())):
                tmpstr += '\nin_{i}_scale={sc}, in_{i}_zero_point={zp}'.format(i=idx, sc=in_scale.item(),
                                                                               zp=in_zp.item())
            tmpstr += '\nout_scale={sc}, out_zero_point={zp}'.format(sc=self.output_scale.item(),
                                                                     zp=self.output_zero_point.item())
        return tmpstr


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
        clip_acts (ClipNode): See RangeLinearQuantWrapper
        per_channel_wts (bool): Enable quantization of weights using separate quantization parameters per
            output channel
        activation_stats (dict): See RangeLinearQuantWrapper
        clip_n_stds (int): See RangeLinearQuantWrapper
        scale_approx_mult_bits (int): See RangeLinearQuantWrapper
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE, per_channel_wts=False, activation_stats=None,
                 clip_n_stds=None, scale_approx_mult_bits=None):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum, mode,
                                                                clip_acts, activation_stats, clip_n_stds,
                                                                scale_approx_mult_bits)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D and Linear modules')

        self.num_bits_params = num_bits_params
        self.per_channel_wts = per_channel_wts

        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            num_bits_params, signed=mode != LinearQuantMode.ASYMMETRIC_UNSIGNED)

        # Quantize weights - overwrite FP32 weights
        w_scale, w_zero_point = _get_quant_params_from_tensor(wrapped_module.weight, num_bits_params, self.mode,
                                                              per_channel=per_channel_wts)

        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_zero_point', w_zero_point)
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point, self.params_min_q_val,
                              self.params_max_q_val, inplace=True)

        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        device = self.w_scale.device

        if self.preset_act_stats:
            self.in_0_scale = self.in_0_scale.to(device)
            self.register_buffer('accum_scale', self.in_0_scale * self.w_scale)
            if self.per_channel_wts:
                self.accum_scale = self.accum_scale.squeeze(dim=-1)
        else:
            self.accum_scale = 1

        # Quantize bias
        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        if self.has_bias:
            if self.preset_act_stats:
                linear_quantize_clamp(wrapped_module.bias.data, self.accum_scale.squeeze(), 0,
                                      self.accum_min_q_val, self.accum_max_q_val, inplace=True)
            else:
                b_scale, b_zero_point = _get_quant_params_from_tensor(wrapped_module.bias, num_bits_params, self.mode)
                self.register_buffer('b_scale', b_scale)
                self.register_buffer('b_zero_point', b_zero_point)
                base_b_q = linear_quantize_clamp(wrapped_module.bias.data, self.b_scale, self.b_zero_point,
                                                 self.params_min_q_val, self.params_max_q_val)
                # Dynamic ranges - save in auxiliary buffer, requantize each time based on dynamic input scale factor
                self.register_buffer('base_b_q', base_b_q)

    def get_inputs_quantization_params(self, input):
        if not self.preset_act_stats:
            self.in_0_scale, self.in_0_zero_point = _get_quant_params_from_tensor(
                input, self.num_bits_acts, self.mode, clip=self.clip_acts,
                num_stds=self.clip_n_stds, scale_approx_mult_bits=self.scale_approx_mult_bits)
        return [self.in_0_scale], [self.in_0_zero_point]

    def quantized_forward(self, input_q):
        # See class documentation for quantized calculation details.

        if not self.preset_act_stats:
            self.accum_scale = self.in_0_scale * self.w_scale
            if self.per_channel_wts:
                self.accum_scale = self.accum_scale.squeeze(dim=-1)

            if self.has_bias:
                # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
                bias_requant_scale = self.accum_scale.squeeze() / self.b_scale
                if self.scale_approx_mult_bits is not None:
                    bias_requant_scale = approx_scale_as_mult_and_shift(bias_requant_scale, self.scale_approx_mult_bits)
                self.wrapped_module.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                      bias_requant_scale, 0,
                                                                      self.accum_min_q_val, self.accum_max_q_val)

        # Note the main terms within the summation is:
        #   (x_q + zp_x) * (w_q + zp_w)
        # In a performance-optimized solution, we would expand the parentheses and perform the computation similar
        # to what is described here:
        #   https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
        # However, for now we're more concerned with simplicity rather than speed. So we'll just add the zero points
        # to the input and weights and pass those to the wrapped model. Functionally, since at this point we're
        # dealing solely with integer values, the results are the same either way.

        if self.mode != LinearQuantMode.SYMMETRIC:
            input_q += self.in_0_zero_point
            self.wrapped_module.weight.data += self.w_zero_point

        accum = self.wrapped_module.forward(input_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        if self.mode != LinearQuantMode.SYMMETRIC:
            self.wrapped_module.weight.data -= self.w_zero_point
        return accum

    def get_output_quantization_params(self, accumulator):
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point

        y_f = accumulator / self.accum_scale
        return _get_quant_params_from_tensor(y_f, self.num_bits_acts, self.mode, clip=self.clip_acts,
                                             num_stds=self.clip_n_stds,
                                             scale_approx_mult_bits=self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point

    def extra_repr(self):
        tmpstr = 'mode={0}, '.format(str(self.mode).split('.')[1])
        tmpstr += 'num_bits_acts={0}, num_bits_params={1}, num_bits_accum={2}, '.format(self.num_bits_acts,
                                                                                        self.num_bits_params,
                                                                                        self.num_bits_accum)
        tmpstr += 'clip_acts={0}, '.format(_enum_to_str(self.clip_acts))
        if self.clip_acts == ClipMode.N_STD:
            tmpstr += 'num_stds={} '.format(self.clip_n_stds)
        tmpstr += 'per_channel_wts={}, scale_approx_mult_bits={}'.format(self.per_channel_wts,
                                                                         self.scale_approx_mult_bits)
        tmpstr += '\npreset_activation_stats={0}'.format(self.preset_act_stats)
        if self.per_channel_wts:
            tmpstr += '\nw_scale=PerCh, w_zero_point=PerCh'
        else:
            tmpstr += '\nw_scale={0:.4f}, w_zero_point={1:.4f}'.format(self.w_scale.item(), self.w_zero_point.item())
        if self.preset_act_stats:
            tmpstr += '\nin_scale={0:.4f}, in_zero_point={1:.4f}'.format(self.in_0_scale.item(),
                                                                         self.in_0_zero_point.item())
            tmpstr += '\nout_scale={0:.4f}, out_zero_point={1:.4f}'.format(self.output_scale.item(),
                                                                           self.output_zero_point.item())
        elif self.has_bias:
            tmpstr += '\nbase_b_scale={0:.4f}, base_b_zero_point={1:.4f}'.format(self.b_scale.item(),
                                                                                 self.b_zero_point.item())
        return tmpstr


class NoStatsError(NotImplementedError):
    pass


class RangeLinearQuantConcatWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, scale_approx_mult_bits=None):
        if not isinstance(wrapped_module, distiller.modules.Concat):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.Concat modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantConcatWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                            clip_acts=clip_acts, activation_stats=activation_stats,
                                                            clip_n_stds=clip_n_stds,
                                                            scale_approx_mult_bits=scale_approx_mult_bits)

        if self.preset_act_stats:
            # For concatenation to make sense, we need to match all the inputs' scales, so we
            # set a re-scale factor based on the preset output scale
            for idx, in_scale in enumerate(self.inputs_scales()):
                requant_scale = self.output_scale / in_scale
                if self.scale_approx_mult_bits is not None:
                    requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
                self.register_buffer('in_{0}_requant_scale'.format(idx), requant_scale)

    def inputs_requant_scales(self):
        if not self.preset_act_stats:
            raise RuntimeError('Input quantization parameter iterators only available when activation stats were given')
        for idx in range(self.num_inputs):
            name = 'in_{0}_requant_scale'.format(idx)
            yield getattr(self, name)

    def get_inputs_quantization_params(self, *inputs):
        return self.inputs_scales(), self.inputs_zero_points()

    def quantized_forward(self, *inputs_q):
        # Re-quantize all inputs based to the same range (the output range)
        inputs_re_q = [linear_quantize_clamp(input_q + zp, requant_scale, self.output_zero_point,
                                             self.acts_min_q_val, self.acts_max_q_val, inplace=False)
                       for input_q, requant_scale, zp in zip(inputs_q, self.inputs_requant_scales(),
                                                             self.inputs_zero_points())]
        return self.wrapped_module(*inputs_re_q)

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        # Nothing to do here, since we already re-quantized in quantized_forward prior to the actual concatenation
        return 1., 0.


class RangeLinearQuantEltwiseAddWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, scale_approx_mult_bits=None):
        if not isinstance(wrapped_module, distiller.modules.EltwiseAdd):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.EltwiseAdd modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantEltwiseAddWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                                clip_acts=clip_acts, activation_stats=activation_stats,
                                                                clip_n_stds=clip_n_stds,
                                                                scale_approx_mult_bits=scale_approx_mult_bits)

        if self.preset_act_stats:
            # For addition to make sense, all input scales must match. So we set a re-scale factor according
            # to the preset output scale
            requant_scales = [self.output_scale / in_scale for in_scale in self.inputs_scales()]
            if scale_approx_mult_bits is not None:
                requant_scales = [approx_scale_as_mult_and_shift(requant_scale, scale_approx_mult_bits)
                                  for requant_scale in requant_scales]
            for idx, requant_scale in enumerate(requant_scales):
                self.register_buffer('in_{0}_requant_scale'.format(idx), requant_scale)

    def inputs_requant_scales(self):
        if not self.preset_act_stats:
            raise RuntimeError('Input quantization parameter iterators only available when activation stats were given')
        for idx in range(self.num_inputs):
            name = 'in_{0}_requant_scale'.format(idx)
            yield getattr(self, name)

    def get_inputs_quantization_params(self, *inputs):
        return self.inputs_scales(), self.inputs_zero_points()

    def quantized_forward(self, *inputs_q):
        # Re-scale inputs to the accumulator range
        inputs_re_q = [linear_quantize_clamp(input_q + zp, requant_scale, 0,
                                             self.accum_min_q_val, self.accum_max_q_val, inplace=False)
                       for input_q, requant_scale, zp in zip(inputs_q, self.inputs_requant_scales(),
                                                             self.inputs_zero_points())]
        accum = self.wrapped_module(*inputs_re_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        return accum

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return 1., self.output_zero_point


class RangeLinearQuantEltwiseMultWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, scale_approx_mult_bits=None):
        if not isinstance(wrapped_module, distiller.modules.EltwiseMult):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.EltwiseMult modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantEltwiseMultWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                                 clip_acts=clip_acts, activation_stats=activation_stats,
                                                                 clip_n_stds=clip_n_stds,
                                                                 scale_approx_mult_bits=scale_approx_mult_bits)

        if self.preset_act_stats:
            self.register_buffer('accum_scale', reduce(lambda x, y: x * y, self.inputs_scales()))

    def get_inputs_quantization_params(self, *inputs):
        return self.inputs_scales(), self.inputs_zero_points()

    def quantized_forward(self, *inputs_q):
        if self.mode != LinearQuantMode.SYMMETRIC:
            for input_q, zp in zip(inputs_q, self.inputs_zero_points()):
                input_q += zp

        accum = self.wrapped_module(*inputs_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        return accum

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point


class FP16Wrapper(nn.Module):
    """
    A wrapper that replaces a module with a half precision version.

    Args:
        module (nn.Module): The module to be replaced.
        convert_input (:obj:`bool`, optional): Specifies whether an input conversion
            to fp16 is required for forward. Default: True.
        return_fp32 (:obj:`bool`, optional): Specifies whether the output needs
            to be converted back to fp32. Default: True.
    """
    def __init__(self, module: nn.Module, convert_input=True, return_fp32=True):
        super(FP16Wrapper, self).__init__()
        self.wrapped_module = module.half()
        self.return_fp32 = return_fp32
        self.convert_input_fp16 = convert_input

    def forward(self, *input):
        if self.convert_input_fp16:
            input = distiller.convert_tensors_recursively_to(input, dtype=torch.float16)

        result = self.wrapped_module(*input)
        if self.return_fp32:
            return distiller.convert_tensors_recursively_to(result, dtype=torch.float32)

        return result


class RangeLinearEmbeddingWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, mode=LinearQuantMode.SYMMETRIC, stats=None):
        if not isinstance(wrapped_module, nn.Embedding):
            raise ValueError(self.__class__.__name__ + ' can only wrap torch.nn.Embedding modules')

        super(RangeLinearEmbeddingWrapper, self).__init__()

        self.min_q_val, self.max_q_val = get_quantized_range(num_bits,
                                                             signed=mode != LinearQuantMode.ASYMMETRIC_UNSIGNED)

        if stats is None:
            w_scale, w_zero_point = _get_quant_params_from_tensor(wrapped_module.weight, num_bits, self.mode)
        else:
            w_scale, w_zero_point = _get_quant_params_from_stats_dict(stats['output'], num_bits, mode)

        device = wrapped_module.weight.device

        self.register_buffer('w_scale', w_scale.to(device))
        self.register_buffer('w_zero_point', w_zero_point.to(device))
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point, self.min_q_val,
                              self.max_q_val, inplace=True)

        self.wrapped_module = wrapped_module

    def forward(self, input):
        out_q = self.wrapped_module(input)
        out_f = linear_dequantize(out_q, self.w_scale, self.w_zero_point, inplace=True)
        return out_f


class PostTrainLinearQuantizer(Quantizer):
    """
    Applies range-based linear quantization to a model.
    This quantizer is expected to be executed at evaluation only, on a pre-trained model
    Currently, the following Modules are supported: torch.nn.Conv2d, torch.nn.Linear

    Args:
        model (torch.nn.Module): Model to be quantized
        bits_activations/parameters/accum (int): Number of bits to be used when quantizing each tensor type
        overrides (:obj:`OrderedDict`, optional): Overrides the layers quantization settings.
        mode (LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed / unsigned)
        clip_acts (ClipMode): Activations clipping mode to use
        per_channel_wts (bool): Enable quantization of weights using separate quantization parameters per
            output channel
        model_activation_stats (str / dict / OrderedDict): Either a path to activation stats YAML file, or a dictionary
            containing the stats. The stats are used for static calculation of quantization parameters.
            The dict should be in the format exported by distiller.data_loggers.QuantCalibrationStatsCollector.
            If None then parameters are calculated dynamically.
        fp16 (bool): Set to True to convert modules to half precision.
        clip_n_stds (float): When clip_acts == ClipMode.N_STD, this is the number of standard deviations to use
        scale_approx_mult_bits (int): If not None, scale factors will be approximated using an integer multiplication
            followed by a bit-wise shift. This eliminates floating-point scale factors, replacing them with integer
            calculations.
            If None, scale factors will be kept in their original FP32 values.
    Note:
        If fp16 is set to True, all the layers (except those overriden in `overrides`) will be converted
        to half precision, regardless of bits_activations/parameters/accum.
    """
    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 per_channel_wts=False, model_activation_stats=None, fp16=False, clip_n_stds=None,
                 scale_approx_mult_bits=None):
        super(PostTrainLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters, bits_bias=bits_accum,
                                                       overrides=overrides, train_with_fp_copy=False)

        mode = verify_quant_mode(mode)
        clip_acts = verify_clip_mode(clip_acts)
        if clip_acts == ClipMode.N_STD and clip_n_stds is None:
            raise ValueError('clip_n_stds must not be None when clip_acts set to N_STD')

        if model_activation_stats is not None:
            if isinstance(model_activation_stats, str):
                if not os.path.isfile(model_activation_stats):
                    raise ValueError("Model activation stats file not found at: " + model_activation_stats)
                msglogger.info('Loading activation stats from: ' + model_activation_stats)
                with open(model_activation_stats, 'r') as stream:
                    model_activation_stats = distiller.utils.yaml_ordered_load(stream)
            elif not isinstance(model_activation_stats, (dict, OrderedDict)):
                raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')

        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters,
                                                    'bits_accum': bits_accum,
                                                    'mode': str(mode).split('.')[1],
                                                    'clip_acts': _enum_to_str(clip_acts),
                                                    'clip_n_stds': clip_n_stds,
                                                    'per_channel_wts': per_channel_wts,
                                                    'fp16': fp16,
                                                    'scale_approx_mult_bits': scale_approx_mult_bits}}

        def replace_param_layer(module, name, qbits_map, per_channel_wts=per_channel_wts,
                                mode=mode, fp16=fp16, scale_approx_mult_bits=scale_approx_mult_bits,
                                clip_acts=clip_acts, clip_n_stds=clip_n_stds):
            if fp16:
                return FP16Wrapper(module)
            norm_name = distiller.utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts)
            return RangeLinearQuantParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                                     num_bits_accum=self.bits_accum, mode=mode, clip_acts=clip_acts,
                                                     per_channel_wts=per_channel_wts,
                                                     activation_stats=self.model_activation_stats.get(norm_name, None),
                                                     clip_n_stds=clip_n_stds,
                                                     scale_approx_mult_bits=scale_approx_mult_bits)

        def replace_non_param_layer(wrapper_type, module, name, qbits_map, fp16=fp16,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    clip_acts=clip_acts, clip_n_stds=clip_n_stds):
            if fp16:
                return FP16Wrapper(module)
            norm_name = distiller.utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts)
            try:
                return wrapper_type(module, qbits_map[name].acts, mode=mode, clip_acts=clip_acts,
                                    activation_stats=self.model_activation_stats.get(norm_name, None),
                                    clip_n_stds=clip_n_stds, scale_approx_mult_bits=scale_approx_mult_bits)
            except NoStatsError:
                msglogger.warning('WARNING: {0} - quantization of {1} without stats not supported. '
                                  'Keeping the original FP32 module'.format(name, module.__class__.__name__))
                return module

        def replace_embedding(module, name, qbits_map, fp16=fp16):
            if fp16:
                return FP16Wrapper(module, convert_input=False)
            norm_name = distiller.utils.normalize_module_name(name)
            return RangeLinearEmbeddingWrapper(module, qbits_map[name].wts, mode=mode,
                                               stats=self.model_activation_stats.get(norm_name, None))

        self.clip_acts = clip_acts
        self.clip_n_stds = clip_n_stds
        self.model_activation_stats = model_activation_stats or {}
        self.bits_accum = bits_accum
        self.mode = mode

        self.replacement_factory[nn.Conv2d] = replace_param_layer
        self.replacement_factory[nn.Linear] = replace_param_layer

        self.replacement_factory[distiller.modules.Concat] = partial(
            replace_non_param_layer, RangeLinearQuantConcatWrapper)
        self.replacement_factory[distiller.modules.EltwiseAdd] = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseAddWrapper)
        self.replacement_factory[distiller.modules.EltwiseMult] = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseMultWrapper)
        self.replacement_factory[nn.Embedding] = replace_embedding

        self.save_per_layer_parameters(msglogger.logdir)

    @classmethod
    def from_args(cls, model, args):
        """
        Returns an instance of PostTrainLinearQuantizer based on the set command-line arguments that are
        given by add_post_train_quant_args()
        """
        if args.qe_config_file:
            return distiller.config_component_from_file_by_class(model, args.qe_config_file,
                                                                 'PostTrainLinearQuantizer')
        else:
            overrides = OrderedDict(
                [(layer, OrderedDict([('clip_acts', 'NONE')]))
                 for layer in args.qe_no_clip_layers]
            )
            return cls(model,
                       bits_activations=args.qe_bits_acts,
                       bits_parameters=args.qe_bits_wts,
                       bits_accum=args.qe_bits_accum,
                       mode=args.qe_mode,
                       clip_acts=args.qe_clip_acts,
                       per_channel_wts=args.qe_per_channel,
                       model_activation_stats=args.qe_stats_file,
                       clip_n_stds=args.qe_clip_n_stds,
                       scale_approx_mult_bits=args.qe_scale_approx_bits,
                       overrides=overrides)

    def save_per_layer_parameters(self, save_dir=''):
        defaults = OrderedDict(self.model.quantizer_metadata['params'])
        defaults.pop('bits_activations')
        defaults.pop('bits_parameters')
        defaults.pop('bits_accum')
        out = OrderedDict()
        for n, m in self.model.named_modules():
            if distiller.has_children(m):
                continue
            qbits = self.module_qbits_map[n]
            d = OrderedDict()
            d['bits_activations'] = qbits.acts
            d['bits_weights'] = qbits.wts
            d['bits_bias'] = qbits.bias
            for k, v in defaults.items():
                actual_v = self.module_overrides_map[n].get(k, v)
                d[k] = actual_v
            out[n] = d
        save_path = os.path.join(save_dir, 'layer_quant_params.yaml')
        distiller.yaml_ordered_save(save_path, out)
        msglogger.info('Per-layer quantization parameters saved to ' + save_path)


###############################################################################
# Quantization-aware training
###############################################################################


def update_ema(biased_ema, value, decay, step):
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # Bias correction
    return biased_ema, unbiased_ema


def inputs_quantize_wrapped_forward(self, input):
    input = self.inputs_quant(input)
    return self.original_forward(input)


class FakeLinearQuantization(nn.Module):
    def __init__(self, num_bits=8, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True, inplace=False):
        super(FakeLinearQuantization, self).__init__()

        self.num_bits = num_bits
        self.mode = mode
        self.dequantize = dequantize
        self.inplace = inplace

        # We track activations ranges with exponential moving average, as proposed by Jacob et al., 2017
        # https://arxiv.org/abs/1712.05877
        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def forward(self, input):
        # We update the tracked stats only in training
        #
        # Due to the way DataParallel works, we perform all updates in-place so the "main" device retains
        # its updates. (see https://pytorch.org/docs/stable/nn.html#dataparallel)
        # However, as it is now, the in-place update of iter_count causes an error when doing
        # back-prop with multiple GPUs, claiming a variable required for gradient calculation has been modified
        # in-place. Not clear why, since it's not used in any calculations that keep a gradient.
        # It works fine with a single GPU. TODO: Debug...
        if self.training:
            with torch.no_grad():
                current_min, current_max = get_tensor_min_max(input)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_decay,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_decay,
                                                                             self.iter_count)

        if self.mode == LinearQuantMode.SYMMETRIC:
            max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
            actual_min, actual_max = -max_abs, max_abs
            if self.training:
                self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, max_abs)
        else:
            actual_min, actual_max = self.tracked_min, self.tracked_max
            signed = self.mode == LinearQuantMode.ASYMMETRIC_SIGNED
            if self.training:
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min,
                                                                                              self.tracked_max,
                                                                                              signed=signed)

        input = clamp(input, actual_min.item(), actual_max.item(), False)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, False)

        return input

    def extra_repr(self):
        mode_str = str(self.mode).split('.')[1]
        return 'mode={0}, num_bits={1}, ema_decay={2:.4f})'.format(mode_str, self.num_bits, self.ema_decay)


class FakeQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, quant_mode, ema_decay):
        super(FakeQuantizationWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.fake_q = FakeLinearQuantization(num_bits, quant_mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))

    def forward(self, *input):
        res = self.wrapped_module(*input)
        res = self.fake_q(res)
        return res


class QuantAwareTrainRangeLinearQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None):
        super(QuantAwareTrainRangeLinearQuantizer, self).__init__(model, optimizer=optimizer,
                                                                  bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_bias=bits_bias,
                                                                  overrides=overrides,
                                                                  train_with_fp_copy=True)

        mode = verify_quant_mode(mode)

        self.model.quantizer_metadata['params']['mode'] = str(mode).split('.')[1]
        self.model.quantizer_metadata['params']['ema_decay'] = ema_decay
        self.model.quantizer_metadata['params']['per_channel_wts'] = per_channel_wts
        self.model.quantizer_metadata['params']['quantize_inputs'] = quantize_inputs

        # Keeping some parameters for input quantization
        self.quantize_inputs = quantize_inputs
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.mode = mode
        self.decay = ema_decay
        self.per_channel_wts = per_channel_wts

        def linear_quantize_param(param_fp, param_meta):
            m = param_meta.module
            # We don't quantize the learned weights of embedding layers per-channel, because they're used
            # as inputs in subsequent layers and we don't support per-channel activations quantization yet
            perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]

            with torch.no_grad():
                scale, zero_point = _get_quant_params_from_tensor(param_fp, param_meta.num_bits, mode,
                                                                  per_channel=perch)
            setattr(m, param_meta.q_attr_name + '_scale', scale)
            setattr(m, param_meta.q_attr_name + '_zero_point', zero_point)
            out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, True, False)
            return out

        def activation_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, bits_acts, mode, ema_decay)

        self.param_quantization_fn = linear_quantize_param

        self.activation_replace_fn = activation_replace_fn
        self.replacement_factory[nn.ReLU] = self.activation_replace_fn

    def _prepare_model_impl(self):
        super(QuantAwareTrainRangeLinearQuantizer, self)._prepare_model_impl()

        if self.quantize_inputs:
            if isinstance(self.model, nn.DataParallel):
                m = self.model.module
            else:
                m = self.model

            m.inputs_quant = FakeLinearQuantization(self.num_bits_inputs, self.mode, self.decay,
                                                    dequantize=True, inplace=False)
            m.__class__.original_forward = m.__class__.forward
            m.__class__.forward = inputs_quantize_wrapped_forward

        # Prepare scale and zero point buffers in modules where parameters are being quantized
        # We're calculating "dummy" scale and zero point just to get their dimensions
        for ptq in self.params_to_quantize:
            m = ptq.module
            param_fp = getattr(m, ptq.fp_attr_name)
            perch = not isinstance(m, nn.Embedding) and self.per_channel_wts and param_fp.dim() in [2, 4]
            with torch.no_grad():
                scale, zero_point = _get_quant_params_from_tensor(param_fp, ptq.num_bits, self.mode,
                                                                  per_channel=perch)
            m.register_buffer(ptq.q_attr_name + '_scale', torch.ones_like(scale))
            m.register_buffer(ptq.q_attr_name + '_zero_point', torch.zeros_like(zero_point))
