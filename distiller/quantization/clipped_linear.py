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
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


class WRPNQuantizer(Quantizer):
    def __init__(self, model, bits_activations=32, bits_weights=32, bits_overrides={}):
        super(WRPNQuantizer, self).__init__(model, bits_activations=bits_activations, bits_weights=bits_weights,
                                            bits_overrides=bits_overrides)

        def wrpn_quantize_param(param_fp, num_bits):
            scale_factor = symmetric_linear_quantization_scale_factor(num_bits, 1)
            param_fp.clamp_(-1, 1)
            param_q = linear_quantize(param_fp, scale_factor, inplace=False)
            linear_dequantize(param_q, scale_factor, inplace=True)
            return param_q

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = wrpn_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn
