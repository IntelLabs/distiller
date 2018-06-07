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
from .level_pruner import SparsityLevelParameterPruner
from distiller.utils import *

import distiller

class BaiduRNNPruner(_ParameterPruner):
    """An element-wise pruner for RNN networks.

    Narang, Sharan & Diamos, Gregory & Sengupta, Shubho & Elsen, Erich. (2017).
    Exploring Sparsity in Recurrent Neural Networks.
    (https://arxiv.org/abs/1704.05119)
    """

    def __init__(self, name, initial_sparsity, final_sparsity, q, ramp_epoch, ramp_slope_mult, weights):
        super(BaiduRNNPruner, self).__init__(name)
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        assert final_sparsity > initial_sparsity
        self.params_names = weights
        assert self.params_names

        self.q = q
        self.ramp_epoch = ramp_epoch
        self.ramp_slope_mult = ramp_slope_mult
        self.ramp_slope = None
        self.start_slope = None

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.params_names:
            return

        starting_epoch = meta['starting_epoch']
        current_epoch = meta['current_epoch']
        ending_epoch = meta['ending_epoch']
        freq = meta['frequency']

        # Calculate start slope
        if self.start_slope is None:
            self.start_slope = (2 * self.q * freq) / (2*(self.ramp_epoch - starting_epoch) + 3*(ending_epoch - self.ramp_epoch))
            self.ramp_slope = self.start_slope * self.ramp_slope_mult

        if current_epoch < self.ramp_epoch:
            eps = self.start_slope * (current_epoch - starting_epoch + 1) / freq
        else:
            eps = (self.start_slope * (self.ramp_epoch - starting_epoch + 1) +
                   self.ramp_slope  * (current_epoch  - self.ramp_epoch + 1)) / freq

        # After computing the threshold, we can create the mask
        zeros_mask_dict[param_name].mask = distiller.threshold_mask(param.data, eps)
