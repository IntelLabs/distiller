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


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LSQLinearQuantization(nn.Module):
    def __init__(self, size, num_bits, dequantize=True, inplace=False, half_range=False):
        super(LSQLinearQuantization, self).__init__()
        self.size = size
        self.num_bits = num_bits
        self.learned_scale_activation = nn.Parameter(torch.tensor([(2 ** num_bits - 1) / 3.]))
        self.dequantize = dequantize
        self.inplace = inplace
        self.half_range = half_range
        self.initialized = False

        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('delta_mse', torch.zeros(1))
        self.register_buffer('scale_init', torch.zeros(1))

    def forward(self, input):
        current_min, current_max = get_tensor_min_max(input)
        self.tracked_min.data = current_min
        self.tracked_max.data = current_max

        if not self.initialized:
            with torch.no_grad():
                clipper = AciqAsymetricClipper(self.num_bits, AciqClipper.AciqClippingType.Laplace,
                                               half_range=self.half_range)

                _, clipped_max = clipper(input)
                rho = 0.5
                self.scale_init.data = (2**self.num_bits - 1) / (rho*clipped_max + (1 - rho)*current_max)
                self.initialized = True

        # Assume relu with zero point = 0

        # Quantize
        input_q = self.learned_scale_activation * input

        # clamp and round
        input_q = torch.clamp(input_q, 0, 2**self.num_bits - 1)
        input_q = RoundSTE.apply(input_q)

        # dequantize
        input_q = input_q / self.learned_scale_activation

        delta = input_q.detach() - input.detach()
        self.delta_mse.data = torch.norm(delta) / delta.numel()

        return input_q

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else 
        return '{0}(num_bits={1}, {2})'.format(self.__class__.__name__, self.num_bits, inplace_str)


class LSQParamsQuantization:
    def __init__(self, per_channel=False):
        self.per_channel = per_channel

    def __call__(self, param_fp, param_meta):
        if self.per_channel:
            out = self.lsq_quantize_param_per_channel(param_fp, param_meta)
        else:
            out = self.lsq_quantize_param(param_fp, param_meta)

        return out

    @staticmethod
    def lsq_quantize_param_per_channel(param_fp, param_meta):
        # return param_fp
        scale = param_meta.module.learned_scale_weight.view(param_fp.shape[0], 1)
        num_bits = param_meta.num_bits

        orig_shape = param_fp.shape
        param_fp = param_fp.view(param_fp.shape[0], -1)

        # Quantize
        param_q = scale * param_fp

        # clamp and round
        lower = -2 ** (num_bits - 1)
        upper = 2 ** (num_bits - 1) - 1
        param_q = torch.clamp(param_q, lower, upper)
        param_q = RoundSTE.apply(param_q)

        # Dequantize
        param_q = param_q / scale

        return param_q.view(orig_shape)

    @staticmethod
    def lsq_quantize_param(param_fp, param_meta):
        # return param_fp
        scale = param_meta.module.learned_scale_weight
        num_bits = param_meta.num_bits

        # Quantize
        param_q = scale * param_fp

        # clamp and round
        lower = -2 ** (num_bits - 1)
        upper = 2 ** (num_bits - 1) - 1
        param_q = torch.clamp(param_q, lower, upper)
        param_q = RoundSTE.apply(param_q)

        # Dequantize
        param_q = param_q / scale

        return param_q

    @staticmethod
    def initialize_scale(float_weight, num_bits, per_channel_wts):
        if per_channel_wts:
            float_weight = float_weight.view(float_weight.shape[0], -1)
            max_ = torch.abs(float_weight.max(-1)[0])
            min_ = torch.abs(float_weight.min(-1)[0])
            rng = torch.max(min_, max_)
            scale = (2**(num_bits - 1) - 1) / rng
        else:
            rng = torch.max(torch.abs(float_weight.max()), torch.abs(float_weight.min()))
            scale = torch.tensor([(2**(num_bits - 1) - 1) / rng])

        return scale


class LSQQuatizer(Quantizer):
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=False, scale_act_decay=None, scale_act_lr=None, scale_w_decay=None, scale_w_lr=None,
                 per_channel_wts=False):
        super(LSQQuatizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_overrides=bits_overrides,
                                            train_with_fp_copy=True, quantize_bias=quantize_bias)

        self.scale_act_decay = scale_act_decay
        self.scale_act_lr = scale_act_lr
        self.scale_w_decay = scale_w_decay
        self.scale_w_lr = scale_w_lr
        self.per_channel_wts = per_channel_wts

        self.initialized = False

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, LSQLinearQuantization(module.num_features, bits_acts, dequantize=True,
                                                    inplace=module.inplace, half_range=True))

        self.replacement_factory[nn.ReLU] = relu_replace_fn

        self.param_quantization_fn = LSQParamsQuantization(per_channel=per_channel_wts)

    def get_loger_stats(self, model, optimizer):
        stats_dict = OrderedDict()
        stats_dict['global/scale_act_lr'] = optimizer.param_groups[1]['lr']
        stats_dict['global/scale_act_decay'] = optimizer.param_groups[1]['weight_decay']
        stats_dict['global/scale_w_lr'] = optimizer.param_groups[2]['lr']
        stats_dict['global/scale_w_decay'] = optimizer.param_groups[2]['weight_decay']
        scale_params = [(n, p) for n, p in model.named_parameters() if 'learned_scale' in n]
        for name, param in scale_params:
            stats_dict[name.replace('module.', ) + '/scale'] = param.item() if param.numel() == 1 else param.mean()
        stats1 = ('Scale/', stats_dict)

        stats_dict = OrderedDict()
        tract_min = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'tracked_min' in k and 'biased' not in k]
        tract_max = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'tracked_max' in k and 'biased' not in k]
        for name, param in tract_min:
            name = name.replace('module.', )
            stats_dict[name] = param.item()
        for name, param in tract_max:
            name = name.replace('module.', )
            stats_dict[name] = param.item()
        stats3 = ('Range/', stats_dict)

        stats_dict = OrderedDict()
        delta = [(k, model.state_dict()[k]) for k in model.state_dict() if
                     'delta_mse' in k]
        for name, param in delta:
            name = name.replace('module.', )
            stats_dict[name] = param.item()
        stats4 = ('Q_delta/', stats_dict)

        return [stats1, stats3, stats4]

    def on_minibatch_end(self, epoch, train_step, steps_per_epoch, optimizer):
        self.quantize_params()

        if not self.initialized:
            scale_init_act = [(k, self.model.state_dict()[k]) for k in self.model.state_dict() if
                         'scale_init' in k]
            scale_params_act = [(n, p) for n, p in self.model.named_parameters() if 'learned_scale_activation' in n]
            for n, p in scale_params_act:
                l_name = n.replace('.learned_scale_activation', )
                scale = [t for n, t in scale_init_act if l_name in n][0]
                learned_scale_param = [p for n, p in scale_params_act if l_name in n][0]
                learned_scale_param.data.copy_(scale)

            self.initialized = True

    def _get_updated_optimizer_params_groups(self):
        base_group = {'params': [param for name, param in self.model.named_parameters() if 'learned_scale' not in name]}
        scale_act_group = {'params': [param for name, param in self.model.named_parameters() if 'learned_scale_activation' in name]}
        scale_w_group = {'params': [param for name, param in self.model.named_parameters() if 'learned_scale_weight' in name]}

        if self.scale_act_lr is not None:
            scale_act_group['lr'] = self.scale_act_lr
        if self.scale_act_decay is not None:
            scale_act_group['weight_decay'] = self.scale_act_decay

        if self.scale_w_lr is not None:
            scale_w_group['lr'] = self.scale_w_lr
        if self.scale_w_decay is not None:
            scale_w_group['weight_decay'] = self.scale_w_decay

        return [base_group, scale_act_group, scale_w_group]

    def _prepare_model_impl(self):
        super(LSQQuatizer, self)._prepare_model_impl()

        for ptq in self.params_to_quantize:
            m = ptq.module
            m.learned_scale_weight = nn.Parameter(LSQParamsQuantization.initialize_scale(
                ptq.module.float_weight, ptq.num_bits, self.per_channel_wts))
