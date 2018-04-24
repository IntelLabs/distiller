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

import logging
import torch
import distiller
from .pruner import _ParameterPruner
msglogger = logging.getLogger()

class L1RankedStructureParameterPruner(_ParameterPruner):
    """Uses mean L1-norm to rank structures and prune a specified percentage of structures
    """
    def __init__(self, name, reg_regims):
        super(L1RankedStructureParameterPruner, self).__init__(name)
        self.name = name
        self.reg_regims = reg_regims

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.reg_regims.keys():
            return

        group_type = self.reg_regims[param_name][1]
        fraction_to_prune = self.reg_regims[param_name][0]
        if fraction_to_prune == 0:
            return

        assert group_type == "3D", "Currently only filter ranking is supported"
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        view_filters = param.view(param.size(0), -1)
        filter_mags = view_filters.data.abs().mean(dim=1)
        topk_filters = int(fraction_to_prune * filter_mags.size(0))
        if topk_filters == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        bottomk, _ = torch.topk(filter_mags, topk_filters, largest=False, sorted=True)
        threshold = bottomk[-1]
        binary_map = filter_mags.gt(threshold).type(type(param.data))
        expanded = binary_map.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
        zeros_mask_dict[param_name].mask = expanded.view(param.size(0), param.size(1), param.size(2), param.size(3))
        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                       distiller.sparsity(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, topk_filters, filter_mags.size(0))
