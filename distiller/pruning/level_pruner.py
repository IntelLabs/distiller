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

import torch
from .pruner import _ParameterPruner
import distiller

class SparsityLevelParameterPruner(_ParameterPruner):
    """Prune to an exact pruning level specification.

    This pruner is very similar to MagnitudeParameterPruner, but instead of
    specifying an absolute threshold for pruning, you specify a target sparsity
    level (expressed as a fraction: 0.5 means 50% sparsity.)

    To find the correct threshold, we view the tensor as one large 1D vector, sort
    it using the absolute values of the elements, and then take topk elements.
    """

    def __init__(self, name, levels, **kwargs):
        super(SparsityLevelParameterPruner, self).__init__(name)
        self.levels = levels
        assert self.levels

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        # If there is a specific sparsity level specified for this module, then
        # use it.  Otherwise try to use the default level ('*').
        desired_sparsity = self.levels.get(param_name, self.levels.get('*', 0))
        if desired_sparsity == 0:
            return

        zeros_mask_dict[param_name].mask = SparsityLevelParameterPruner.create_mask(param, desired_sparsity)

    @staticmethod
    def create_mask(param, desired_sparsity):
        with torch.no_grad():
            bottomk, _ = torch.topk(param.abs().view(-1), int(desired_sparsity * param.numel()), largest=False, sorted=True)
            threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
            mask = distiller.threshold_mask(param.data, threshold)
            return mask