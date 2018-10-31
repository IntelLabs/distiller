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
import distiller
import torch
import logging
msglogger = logging.getLogger()


class SplicingPruner(_ParameterPruner):
    """A pruner that allows both prunes and splices connections

    The idea of pruning and splicing working in tandem was first proposed in the following
    NIPS paper from Intel Labs China in 2016:
        Dynamic Network Surgery for Efficient DNNs, Yiwen Guo, Anbang Yao, Yurong Chen.
        NIPS 2016, https://arxiv.org/abs/1608.04493.

    A SplicingPruner works best with a Dynamic Network Surgery schedule.
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

        if self.sensitivity_multiplier > 0:
            # Linearly growing sensitivity - for now this is hard-coded
            starting_epoch = meta['starting_epoch']
            current_epoch = meta['current_epoch']
            sensitivity *= (current_epoch - starting_epoch) * self.sensitivity_multiplier + 1

        if zeros_mask_dict[param_name].mask is None:
            zeros_mask_dict[param_name].mask = torch.ones_like(param)
        masked_weights = param.mul(zeros_mask_dict[param_name].mask)
        mean = torch.mean(masked_weights).item()
        std = torch.std(masked_weights).item()

        # After computing the threshold, we can create the mask
        threshold_low = (mean + std * sensitivity) * 0.9
        threshold_hi = (mean + std * sensitivity) * 1.1
        a = distiller.threshold_mask(param.data, threshold_low)
        b = a.mul_(zeros_mask_dict[param_name].mask)

        #msglogger.info("{}: to threshold={} : {}  mean={}  std={}".format(param_name, threshold_low, threshold_hi, mean, std))
        zeros_mask_dict[param_name].mask = torch.clamp(b + distiller.threshold_mask(param.data, threshold_hi), 0, 1)
        #msglogger.info("sparsity of {}  = {}".format(param_name, distiller.sparsity(zeros_mask_dict[param_name].mask)))
