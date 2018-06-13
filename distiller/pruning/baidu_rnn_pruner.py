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

    This implementation slightly differs from the algorithm original paper in that
    the algorithm changes the pruning rate at the training-step granularity, while
    Distiller controls the pruning rate at epoch granularity.

    Equation (1):

                                            2 * q * freq
        start_slope = -------------------------------------------------------
                      2 * (ramp_itr - start_itr ) + 3 * (end_itr - ramp_itr )


    Pruning algorithm (1):

        if current itr < ramp itr then
                threshold =  start_slope * (current_itr - start_itr + 1) / freq
        else
                threshold = (start_slope * (ramp_itr    - start_itr + 1) +
                             ramp_slope  * (current_itr - ramp_itr  + 1)) / freq
         end if

         mask = abs(param) < threshold
    """

    def __init__(self, name, q, ramp_epoch_offset, ramp_slope_mult, weights):
        # Initialize the pruner, using a configuration that originates from the
        # schedule YAML file.
        super(BaiduRNNPruner, self).__init__(name)
        self.params_names = weights
        assert self.params_names

        # This is the 'q' value that appears in equation (1) of the paper
        self.q = q
        # This is the number of epochs to wait after starting_epoch, before we
        # begin ramping up the pruning rate.
        # In other words, between epochs 'starting_epoch' and 'starting_epoch'+
        # self.ramp_epoch_offset the pruning slope is 'self.start_slope'.  After
        # that, the slope is 'self.ramp_slope'
        self.ramp_epoch_offset = ramp_epoch_offset
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

        ramp_epoch = self.ramp_epoch_offset + starting_epoch

        # Calculate start slope
        if self.start_slope is None:
            # We want to calculate these values only once, and then cache them.
            self.start_slope = (2 * self.q * freq) / (2*(ramp_epoch - starting_epoch) + 3*(ending_epoch - ramp_epoch))
            self.ramp_slope = self.start_slope * self.ramp_slope_mult

        if current_epoch < ramp_epoch:
            eps = self.start_slope * (current_epoch - starting_epoch + 1) / freq
        else:
            eps = (self.start_slope * (ramp_epoch - starting_epoch + 1) +
                   self.ramp_slope  * (current_epoch  - ramp_epoch + 1)) / freq

        # After computing the threshold, we can create the mask
        zeros_mask_dict[param_name].mask = distiller.threshold_mask(param.data, eps)
