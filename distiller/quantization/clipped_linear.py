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

from collections import OrderedDict
import torch.nn as nn

from .quantizer import Quantizer
from .q_utils import *
import logging
msglogger = logging.getLogger()

###
# Clipping-based linear quantization (e.g. DoReFa, WRPN)
###


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale_factor, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None


class ClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super(ClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, clip_val)
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        input = LinearQuantizeSTE.apply(input, self.scale_factor, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else 
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


class WRPNQuantizer(Quantizer):
    """
    Quantizer using the WRPN quantization scheme, as defined in:
    Mishra et al., WRPN: Wide Reduced-Precision Networks (https://arxiv.org/abs/1709.01134)

    Notes:
        1. This class does not take care of layer widening as described in the paper
        2. The paper defines special handling for 1-bit weights which isn't supported here yet
    """
    def __init__(self, model, bits_activations=32, bits_weights=32, bits_overrides=OrderedDict(), quantize_bias=False):
        super(WRPNQuantizer, self).__init__(model, bits_activations=bits_activations, bits_weights=bits_weights,
                                            bits_overrides=bits_overrides, train_with_fp_copy=True,
                                            quantize_bias=quantize_bias)

        def wrpn_quantize_param(param_fp, num_bits):
            scale_factor = symmetric_linear_quantization_scale_factor(num_bits, 1)
            out = param_fp.clamp(-1, 1)
            out = LinearQuantizeSTE.apply(out, scale_factor, True, False)
            return out

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = wrpn_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn


class DorefaQuantizer(Quantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)

    Notes:
        1. Gradients quantization not supported yet
        2. The paper defines special handling for 1-bit weights which isn't supported here yet
    """
    def __init__(self, model, bits_activations=32, bits_weights=32, bits_overrides=OrderedDict(), quantize_bias=False):
        super(DorefaQuantizer, self).__init__(model, bits_activations=bits_activations, bits_weights=bits_weights,
                                              bits_overrides=bits_overrides, train_with_fp_copy=True,
                                              quantize_bias=quantize_bias)

        def dorefa_quantize_param(param_fp, num_bits):
            scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, 1)
            out = param_fp.tanh()
            out = out / (2 * out.abs().max()) + 0.5
            out = LinearQuantizeSTE.apply(out, scale_factor, True, False)
            out = 2 * out - 1
            return out

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = dorefa_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn
