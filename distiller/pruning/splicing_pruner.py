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


from .pruner import _ParameterPruner
import torch
import logging
msglogger = logging.getLogger()


class SplicingPruner(_ParameterPruner):
    """A pruner that both prunes and splices connections.

    The idea of pruning and splicing working in tandem was first proposed in the following
    NIPS paper from Intel Labs China in 2016:
        Dynamic Network Surgery for Efficient DNNs, Yiwen Guo, Anbang Yao, Yurong Chen.
        NIPS 2016, https://arxiv.org/abs/1608.04493.

    A SplicingPruner works best with a Dynamic Network Surgery schedule.
    The original Caffe code from the authors of the paper is available here:
    https://github.com/yiwenguo/Dynamic-Network-Surgery/blob/master/src/caffe/layers/compress_conv_layer.cpp
    """

    def __init__(self, name, sensitivities, low_thresh_mult, hi_thresh_mult, sensitivity_multiplier=0):
        """Arguments:
        """
        super(SplicingPruner, self).__init__(name)
        self.sensitivities = sensitivities
        self.low_thresh_mult = low_thresh_mult
        self.hi_thresh_mult = hi_thresh_mult
        self.sensitivity_multiplier = sensitivity_multiplier

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.sensitivities:
            if '*' not in self.sensitivities:
                return
            else:
                sensitivity = self.sensitivities['*']
        else:
            sensitivity = self.sensitivities[param_name]

        if not hasattr(param, '_std'):
            # Compute the mean and standard-deviation once, and cache them.
            param._std = torch.std(param.abs()).item()
            param._mean = torch.mean(param.abs()).item()

        if self.sensitivity_multiplier > 0:
            # Linearly growing sensitivity - for now this is hard-coded
            starting_epoch = meta['starting_epoch']
            current_epoch = meta['current_epoch']
            sensitivity *= (current_epoch - starting_epoch) * self.sensitivity_multiplier + 1

        threshold_low = (param._mean + param._std * sensitivity) * self.low_thresh_mult
        threshold_hi = (param._mean + param._std * sensitivity) * self.hi_thresh_mult

        if zeros_mask_dict[param_name].mask is None:
            zeros_mask_dict[param_name].mask = torch.ones_like(param)

        # This code performs the code in equation (3) of the "Dynamic Network Surgery" paper:
        #
        #           0    if a  > |W|
        # h(W) =    mask if a <= |W| < b
        #           1    if b <= |W|
        #
        # h(W) is the so-called "network surgery function".
        # mask is the mask used in the previous iteration.
        # a and b are the low and high thresholds, respectively.
        # We followed the example implementation from Yiwen Guo in Caffe, and used the
        # weight tensor's starting mean and std.
        # This is very similar to the initialization performed by distiller.SensitivityPruner.

        masked_weights = param.mul(zeros_mask_dict[param_name].mask).abs()
        a = masked_weights.ge(threshold_low)
        b = a & zeros_mask_dict[param_name].mask.type(torch.cuda.ByteTensor)
        zeros_mask_dict[param_name].mask = (b | masked_weights.ge(threshold_hi)).type(torch.cuda.FloatTensor)
