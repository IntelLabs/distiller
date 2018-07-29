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


# TODO: support different policies for ranking structures
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

        if group_type not in ['3D', 'Channels']:
            raise ValueError("Currently only filter (3D) and channel ranking is supported")
        if group_type == "3D":
            return self.rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict)
        elif group_type == "Channels":
            return self.rank_prune_channels(fraction_to_prune, param, param_name, zeros_mask_dict)

    @staticmethod
    def rank_channels(fraction_to_prune, param):
        num_filters = param.size(0)
        num_channels = param.size(1)
        kernel_size = param.size(2) * param.size(3)

        # First, reshape the weights tensor such that each channel (kernel) in the original
        # tensor, is now a row in the 2D tensor.
        view_2d = param.view(-1, kernel_size)
        # Next, compute the sums of each kernel
        kernel_sums = view_2d.abs().sum(dim=1)
        # Now group by channels
        k_sums_mat = kernel_sums.view(num_filters, num_channels).t()
        channel_mags = k_sums_mat.mean(dim=1)
        k = int(fraction_to_prune * channel_mags.size(0))
        if k == 0:
            msglogger.info("Too few channels (%d)- can't prune %.1f%% channels",
                            num_channels, 100*fraction_to_prune)
            return None, None

        bottomk, _ = torch.topk(channel_mags, k, largest=False, sorted=True)
        return bottomk, channel_mags


    def rank_prune_channels(self, fraction_to_prune, param, param_name, zeros_mask_dict):
        bottomk_channels, channel_mags = self.rank_channels(fraction_to_prune, param)
        if bottomk_channels is None:
            # Empty list means that fraction_to_prune is too low to prune anything
            return

        num_filters = param.size(0)
        num_channels = param.size(1)

        threshold = bottomk_channels[-1]
        binary_map = channel_mags.gt(threshold).type(param.data.type())
        a = binary_map.expand(num_filters, num_channels)
        c = a.unsqueeze(-1)
        d = c.expand(num_filters, num_channels, param.size(2) * param.size(3)).contiguous()
        zeros_mask_dict[param_name].mask = d.view(num_filters, num_channels, param.size(2), param.size(3))

        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                       distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, len(bottomk_channels), num_channels)


    def rank_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        view_filters = param.view(param.size(0), -1)
        filter_mags = view_filters.data.norm(1, dim=1)  # same as view_filters.data.abs().sum(dim=1)
        topk_filters = int(fraction_to_prune * filter_mags.size(0))
        if topk_filters == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        bottomk, _ = torch.topk(filter_mags, topk_filters, largest=False, sorted=True)
        threshold = bottomk[-1]
        binary_map = filter_mags.gt(threshold).type(param.data.type())
        expanded = binary_map.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
        zeros_mask_dict[param_name].mask = expanded.view(param.size(0), param.size(1), param.size(2), param.size(3))
        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                       distiller.sparsity(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, topk_filters, filter_mags.size(0))
