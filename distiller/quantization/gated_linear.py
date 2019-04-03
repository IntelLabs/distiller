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

import torch.nn as nn
from collections import OrderedDict
from .quantizer import Quantizer
from .q_utils import *
from distiller.quantization.range_linear import LinearQuantMode, SaturationMode, FakeQuantizationWrapper
from distiller.quantization.range_linear import verify_sat_mode, verify_mode, update_ema
import logging
msglogger = logging.getLogger()


class GatedLinearQuantization(nn.Module):
    def __init__(self, size, num_bits=8, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True,
                 inplace=False, half_range=False, act_sat_mode=None, do_decay=True):
        super(GatedLinearQuantization, self).__init__()
        self.size = size
        self.num_bits = num_bits
        self.mode = mode
        self.dequantize = dequantize
        self.inplace = inplace
        self.half_range = half_range
        self.act_sat_mode = act_sat_mode
        self.do_decay = do_decay

        self.q_gate = nn.Parameter(torch.Tensor(self.size))
        self.q_gate.data = 1 * torch.ones(self.size)

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
                if self.act_sat_mode is None:
                    current_min, current_max = get_tensor_min_max(input)
                elif self.act_sat_mode == SaturationMode.AVERAGE:
                    current_min, current_max = get_tensor_avg_min_max(input)
                elif self.act_sat_mode == SaturationMode.LAPLACE:
                    clipper = AciqAsymetricClipper(self.num_bits, AciqClipper.AciqClippingType.Laplace, half_range=self.half_range)
                    # current_min, current_max = clipper(input)
                    clipped_min, clipped_max = clipper(input)
                    sat_factor = 0.5
                    current_min, current_max = get_tensor_min_max(input)
                    current_min, current_max = sat_factor * clipped_min + (1 - sat_factor) * current_min, \
                                               sat_factor * clipped_max + (1 - sat_factor) * current_max
                elif self.act_sat_mode == SaturationMode.GAUSS:
                    clipper = AciqAsymetricClipper(self.num_bits, AciqClipper.AciqClippingType.Gauss, half_range=self.half_range)
                    current_min, current_max = clipper(input)

            self.iter_count += 1
            if self.do_decay:
                self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                                 current_min, self.ema_decay,
                                                                                 self.iter_count)
                self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                                 current_max, self.ema_decay,
                                                                                 self.iter_count)
            else:
                self.tracked_min.data = current_min
                self.tracked_max.data = current_max

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
        input_q = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, False)
        delta = input_q - input

        output = input + (1. - torch.clamp(self.q_gate, 0., 1.).view(1, self.size, 1, 1)) * delta

        return output

    def extra_repr(self):
        mode_str = str(self.mode).split('.')[1]
        sat_mode_str = str(self.act_sat_mode).split('.')[1] if self.act_sat_mode else 'No'
        return 'mode={0}, num_bits={1}, ema_decay={2:.4f}, act_sat_mode={3})'.format(mode_str, self.num_bits,
                                                                                     self.ema_decay, sat_mode_str)


class GatedSTEQuatizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=True, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None, act_sat_mode=None, wts_sat_mode=None):
        super(GatedSTEQuatizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_overrides=bits_overrides,
                                                                  quantize_bias=quantize_bias,
                                                                  train_with_fp_copy=True)

        # if isinstance(model, nn.DataParallel) and len(model.device_ids) > 1:
        #     raise RuntimeError('QuantAwareTrainRangeLinearQuantizer currently does not support running with '
        #                        'multiple GPUs')

        mode = verify_mode(mode)
        self.act_sat_mode = verify_sat_mode(act_sat_mode) if act_sat_mode is not None else None
        self.wts_sat_mode = verify_sat_mode(wts_sat_mode) if wts_sat_mode is not None else None

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

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, GatedLinearQuantization(module.num_features, bits_acts, mode, ema_decay,
                                                 dequantize=True,
                                                 inplace=getattr(module, 'inplace', False),
                                                 half_range=True, act_sat_mode=self.act_sat_mode))

        self.replacement_factory[nn.ReLU] = relu_replace_fn

    def _get_updated_optimizer_params_groups(self):
        base_group = {'params': [param for name, param in self.model.named_parameters() if 'q_gate' not in name]}
        q_val_group = {'params': [param for name, param in self.model.named_parameters() if 'q_gate' in name]}
        q_val_group['weight_decay'] = 0.01
        # q_val_group['lr'] = 0.001
        return [base_group, q_val_group]


# Currently clamp with 0 from the bottom
# TODO: generic version with two parameters clamp_min, clamp_max
class LearnedClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clamp_max):
        ctx.save_for_backward(input, clamp_max)
        output = clamp(input, 0, clamp_max.item())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(0)] = 0
        grad_input[input.ge(clip_val.item())] = 0

        grad_alpha = grad_output.clone()
        grad_alpha[input.lt(clip_val.item())] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input, grad_alpha, None


class LearnedClippedGatedLinearQuantization(nn.Module):
    def __init__(self, size, num_bits, dequantize=True, inplace=False, half_range=False):
        super(LearnedClippedGatedLinearQuantization, self).__init__()
        self.size = size
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([0]))
        self.q_gate = nn.Parameter(torch.Tensor(self.size))
        self.q_gate.data = 1 * torch.ones(self.size)
        self.dequantize = dequantize
        self.inplace = inplace
        self.half_range = half_range
        self.clip_value_initialized = False
        self.init_sat_factor = 0.5

        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('delta_mse', torch.tensor([0]))
        # self.register_buffer('delta_8bit_mse', torch.tensor([0]))

    def forward(self, input):
        current_min, current_max = get_tensor_min_max(input)
        self.tracked_min.data = current_min
        self.tracked_max.data = current_max

        if not self.clip_value_initialized:
            # Initialize using Laplace clipping
            with torch.no_grad():
                clipper = AciqAsymetricClipper(self.num_bits, AciqClipper.AciqClippingType.Laplace,
                                               half_range=self.half_range)

                _, clipped_max = clipper(input)
                _, initial_max = get_tensor_min_max(input)
                initial_clipping_val = self.init_sat_factor * clipped_max + (1 - self.init_sat_factor) * initial_max
                self.clip_val.copy_(torch.tensor([initial_clipping_val]))

            self.clip_value_initialized = True

        # Clamp
        # input = LearnedClamp.apply(input, self.clip_val)
        input = torch.where(input < self.clip_val, input, self.clip_val)

        # Quantize
        scale, zero_point = asymmetric_linear_quantization_params(self.num_bits, 0, self.clip_val.item(), signed=False)
        input_q = LinearQuantizeSTE.apply(input, scale, zero_point, self.dequantize, False)
        delta = input_q - input

        self.delta_mse.data = torch.norm(delta) / delta.numel()
        # self.delta_8bit_mse.data = self._qmse(input_q, input, ref_bits=8)

        output = input + (1. - torch.clamp(self.q_gate, 0., 1.).view(1, self.size, 1, 1)) * delta
        return output

    def _qmse(self, tensor, tensor_high_prec, ref_bits):
        scale, zero_point = asymmetric_linear_quantization_params(ref_bits, tensor_high_prec.min(), tensor_high_prec.max(), signed=False)
        tensor_ref_prec = LinearQuantizeSTE.apply(tensor_high_prec, scale, zero_point, True, False)
        mse = torch.norm(tensor - tensor_ref_prec) / tensor.numel()
        return mse

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else 
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


class GatedPactSTEQuatizer(Quantizer):
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=False, act_clip_decay=None, act_clip_lr=None, q_gate_decay=None, q_gate_lr=None):
        super(GatedPactSTEQuatizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_overrides=bits_overrides,
                                            train_with_fp_copy=True, quantize_bias=quantize_bias)

        self.act_clip_decay = act_clip_decay
        self.act_clip_lr = act_clip_lr
        self.q_gate_decay = q_gate_decay
        self.q_gate_lr = q_gate_lr

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, LearnedClippedGatedLinearQuantization(module.num_features, bits_acts, dequantize=True,
                                                    inplace=module.inplace))

        self.replacement_factory[nn.ReLU] = relu_replace_fn

    def get_loger_stats(self, model, optimizer):
        q_val_params = [(n, p) for n, p in model.named_parameters() if 'q_gate' in n]
        stats_dict = OrderedDict()
        stats_dict['global/LR'] = optimizer.param_groups[1]['lr']
        stats_dict['global/weight_decay'] = optimizer.param_groups[1]['weight_decay']
        for name, param in q_val_params:
            stats_dict[name + '/min'] = param.min()
            stats_dict[name + '/max'] = param.max()
        stats1 = ('Q_gate/', stats_dict)

        stats_dict = OrderedDict()
        stats_dict['global/LR'] = optimizer.param_groups[2]['lr']
        stats_dict['global/weight_decay'] = optimizer.param_groups[2]['weight_decay']
        clip_val_params = [(n, p) for n, p in model.named_parameters() if 'clip_val' in n]
        for name, param in clip_val_params:
            stats_dict[name + '/clip_val'] = param.item()
        stats2 = ('Clip/', stats_dict)

        stats_dict = OrderedDict()
        tract_min = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'tracked_min' in k and 'biased' not in k]
        tract_max = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'tracked_max' in k and 'biased' not in k]
        for name, param in tract_min:
            stats_dict[name] = param.item()
        for name, param in tract_max:
            stats_dict[name] = param.item()
        stats3 = ('Range/', stats_dict)

        stats_dict = OrderedDict()
        delta = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'delta_mse' in k]
        for name, param in delta:
            stats_dict[name] = param.item()
        # delta_8bit = [(k, model.state_dict()[k]) for k in model.state_dict() if
        #          'delta_8bit_mse' in k]
        # for name, param in delta_8bit:
        #     stats_dict[name] = param.item()
        stats4 = ('Q_delta/', stats_dict)

        return [stats1, stats2, stats3, stats4]

    def on_minibatch_end(self, epoch, train_step, steps_per_epoch, optimizer):
        pass

    def _get_updated_optimizer_params_groups(self):
        base_group = {'params': [param for name, param in self.model.named_parameters() if 'q_gate' not in name and 'clip_val' not in name]}
        q_gate_group = {'params': [param for name, param in self.model.named_parameters() if 'q_gate' in name]}
        clip_val_group = {'params': [param for name, param in self.model.named_parameters() if 'clip_val' in name]}

        if self.q_gate_lr is not None:
            q_gate_group['lr'] = self.q_gate_lr
        if self.q_gate_decay is not None:
            q_gate_group['weight_decay'] = self.q_gate_decay

        if self.act_clip_lr is not None:
            clip_val_group['lr'] = self.act_clip_lr
        if self.act_clip_decay is not None:
            clip_val_group['weight_decay'] = self.act_clip_decay

        return [base_group, q_gate_group, clip_val_group]
