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
        self.learned_scale = nn.Parameter(torch.tensor([(2**num_bits - 1)/3.]))  # TODO: change to better initialization
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
                self.scale_init.data = (2**self.num_bits - 1) / (0.5*clipped_max + 0.5*current_max)
                self.initialized = True

        # Assume relu with zero point = 0

        # Quantize
        input_q = self.learned_scale * input

        # clamp and round
        input_q = torch.clamp(input_q, 0, 2**self.num_bits - 1)
        input_q = RoundSTE.apply(input_q)

        # dequantize
        input_q = input_q / self.learned_scale

        delta = input_q.detach() - input.detach()
        self.delta_mse.data = torch.norm(delta) / delta.numel()

        return input_q

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else 
        return '{0}(num_bits={1}, {2})'.format(self.__class__.__name__, self.num_bits, inplace_str)


class LSQQuatizer(Quantizer):
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=False, scale_decay=None, scale_lr=None):
        super(LSQQuatizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_overrides=bits_overrides,
                                            train_with_fp_copy=True, quantize_bias=quantize_bias)

        self.scale_decay = scale_decay
        self.scale_lr = scale_lr
        self.initialized = False

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, LSQLinearQuantization(module.num_features, bits_acts, dequantize=True,
                                                    inplace=module.inplace, half_range=True))

        self.replacement_factory[nn.ReLU] = relu_replace_fn

    def get_loger_stats(self, model, optimizer):
        stats_dict = OrderedDict()
        stats_dict['global/LR'] = optimizer.param_groups[1]['lr']
        stats_dict['global/weight_decay'] = optimizer.param_groups[1]['weight_decay']
        scale_params = [(n, p) for n, p in model.named_parameters() if 'learned_scale' in n]
        for name, param in scale_params:
            stats_dict[name.replace('module.', ) + '/scale'] = param.item()
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
        if not self.initialized:
            tract_scale = [(k, self.model.state_dict()[k]) for k in self.model.state_dict() if
                         'scale_init' in k]
            scale_params = [(n, p) for n, p in self.model.named_parameters() if 'learned_scale' in n]
            for n, p in scale_params:
                l_name = n.replace('.learned_scale', )
                scale = [t for n, t in tract_scale if l_name in n][0]
                learned_scale_param = [p for n, p in scale_params if l_name in n][0]
                learned_scale_param.data.copy_(scale)

            self.initialized = True

    def _get_updated_optimizer_params_groups(self):
        base_group = {'params': [param for name, param in self.model.named_parameters() if 'learned_scale' not in name]}
        scale_group = {'params': [param for name, param in self.model.named_parameters() if 'learned_scale' in name]}

        if self.scale_lr is not None:
            scale_group['lr'] = self.scale_lr
        if self.scale_decay is not None:
            scale_group['weight_decay'] = self.scale_decay

        return [base_group, scale_group]
