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

import numpy as np
import logging
import torch
import distiller
from .pruner import _ParameterPruner
msglogger = logging.getLogger()


# TODO: support different policies for ranking structures
class RankedStructureParameterPruner(_ParameterPruner):
    """Uses mean L1-norm to rank structures and prune a specified percentage of structures
    """
    def __init__(self, name, reg_regims):
        super(RankedStructureParameterPruner, self).__init__(name)
        self.reg_regims = reg_regims


class L1RankedStructureParameterPruner(RankedStructureParameterPruner):
    """Uses mean L1-norm to rank structures and prune a specified percentage of structures
    """
    def __init__(self, name, reg_regims):
        super(L1RankedStructureParameterPruner, self).__init__(name, reg_regims)

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.reg_regims.keys():
            return

        group_type = self.reg_regims[param_name][1]
        fraction_to_prune = self.reg_regims[param_name][0]
        if fraction_to_prune == 0:
            return

        if group_type in ['3D', 'Filters']:
            return self.rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict)
        elif group_type == 'Channels':
            return self.rank_prune_channels(fraction_to_prune, param, param_name, zeros_mask_dict)
        elif group_type == 'Rows':
            return self.rank_prune_rows(fraction_to_prune, param, param_name, zeros_mask_dict)
        else:
            raise ValueError("Currently only filter (3D) and channel ranking is supported")

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

    @staticmethod
    def rank_prune_channels(fraction_to_prune, param, param_name, zeros_mask_dict, model=None):
        bottomk_channels, channel_mags = L1RankedStructureParameterPruner.rank_channels(fraction_to_prune, param)
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

    @staticmethod
    def rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model=None):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        # First we rank the filters
        view_filters = param.view(param.size(0), -1)
        filter_mags = view_filters.data.abs().mean(dim=1)
        topk_filters = int(fraction_to_prune * filter_mags.size(0))
        if topk_filters == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return
        bottomk, _ = torch.topk(filter_mags, topk_filters, largest=False, sorted=True)
        threshold = bottomk[-1]
        # Then we threshold
        zeros_mask_dict[param_name].mask = distiller.group_threshold_mask(param, 'Filters', threshold, 'Mean_Abs')
        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                       distiller.sparsity(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, topk_filters, filter_mags.size(0))

    @staticmethod
    def rank_prune_rows(fraction_to_prune, param, param_name, zeros_mask_dict, model=None):
        """Prune the rows of a matrix, based on ranked L1-norms of the matrix rows.

        PyTorch stores the weights matrices in a transposed format.  I.e. before performing GEMM, a matrix is
        transposed.  This is counter-intuitive.  To deal with this, we can either transpose the matrix and
        then proceed to compute the masks as usual, or we can treat columns as rows, and rows as columns :-(.
        We choose the latter, because transposing very large matrices can be detrimental to performance.  Note
        that computing mean L1-norm of columns is also not optimal, because consequtive column elements are far
        away from each other in memory, and this means poor use of caches and system memory.
        """

        assert param.dim() == 2, "This thresholding is only supported for 2D weights"
        ROWS_DIM = 0
        THRESHOLD_DIM = 'Cols'
        rows_mags = param.abs().mean(dim=ROWS_DIM)
        num_rows_to_prune = int(fraction_to_prune * rows_mags.size(0))
        if num_rows_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% rows", 100*fraction_to_prune)
            return
        bottomk_rows, _ = torch.topk(rows_mags, num_rows_to_prune, largest=False, sorted=True)
        threshold = bottomk_rows[-1]
        zeros_mask_dict[param_name].mask = distiller.group_threshold_mask(param, THRESHOLD_DIM, threshold, 'Mean_Abs')
        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                       distiller.sparsity(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_rows_to_prune, rows_mags.size(0))


class RankedFiltersParameterPruner(RankedStructureParameterPruner):
    """Base class for the special (but often-used) case of ranking filters
    """
    def __init__(self, name, reg_regims):
        super(RankedFiltersParameterPruner, self).__init__(name, reg_regims)

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.reg_regims.keys():
            return

        group_type = self.reg_regims[param_name][1]
        fraction_to_prune = self.reg_regims[param_name][0]
        if fraction_to_prune == 0:
            return

        if group_type in ['3D', 'Filters']:
            return self.rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, meta['model'])
        else:
            raise ValueError("Currently only filter (3D) ranking is supported")

    @staticmethod
    def mask_from_filter_order(filters_ordered_by_criterion, param, num_filters):
        binary_map = torch.zeros(num_filters).cuda()
        binary_map[filters_ordered_by_criterion] = 1
        #msglogger.info("binary_map: {}".format(binary_map))
        expanded = binary_map.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
        return expanded.view(param.shape)


class ActivationAPoZRankedFilterPruner(RankedFiltersParameterPruner):
    """Uses mean APoZ (average percentage of zeros) activation channels to rank structures
    and prune a specified percentage of structures.

    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures",
    Hengyuan Hu, Rui Peng, Yu-Wing Tai, Chi-Keung Tang, ICLR 2016
    https://arxiv.org/abs/1607.03250
    """
    def __init__(self, name, reg_regims):
        super(ActivationAPoZRankedFilterParameterPruner, self).__init__(name, reg_regims)

    @staticmethod
    def rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"

        # Use the parameter name to locate the module that has the activation sparsity statistics
        fq_name = param_name.replace(".conv", ".relu")[:-len(".weight")]
        module = distiller.find_module_by_fq_name(model, fq_name)
        if module is None:
            raise ValueError("Could not find a layer named %s in the model."
                             "\nMake sure to use assign_layer_fq_names()" % fq_name)
        if not hasattr(module, 'apoz_channels'):
            raise ValueError("Could not find attribute \'apoz_channels\' in module %s."
                             "\nMake sure to use SummaryActivationStatsCollector(\"apoz_channels\")" % fq_name)

        apoz, std = module.apoz_channels.value()
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)
        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        # Sort from high to low, and remove the bottom 'num_filters_to_prune' filters
        filters_ordered_by_apoz = np.argsort(-apoz)[:-num_filters_to_prune]
        zeros_mask_dict[param_name].mask = RankedFiltersParameterPruner.mask_from_filter_order(filters_ordered_by_apoz,
                                                                                               param, num_filters)

        msglogger.info("ActivationL1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)


class RandomRankedFilterPruner(RankedFiltersParameterPruner):
    """A Random raanking of filters.

    This is used for sanity testing of other algorithms.
    """
    def __init__(self, name, reg_regims):
        super(RandomRankedFilterPruner, self).__init__(name, reg_regims)

    @staticmethod
    def rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)

        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        filters_ordered_randomly = np.random.permutation(num_filters)[:-num_filters_to_prune]
        zeros_mask_dict[param_name].mask = RankedFiltersParameterPruner.mask_from_filter_order(filters_ordered_randomly,
                                                                                               param, num_filters)

        msglogger.info("RandomRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)


class GradientRankedFilterPruner(RankedFiltersParameterPruner):
    """
    """
    def __init__(self, name, reg_regims):
        super(RandomRankedFilterPruner, self).__init__(name, reg_regims)

    @staticmethod
    def rank_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)
        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        # Compute the multiplicatipn of the filters times the filter_gradienrs
        view_filters = param.view(param.size(0), -1)
        view_filter_grads = param.grad.view(param.size(0), -1)
        weighted_gradients = view_filter_grads * view_filters
        weighted_gradients = weighted_gradients.sum(dim=1)

        # Sort from high to low, and remove the bottom 'num_filters_to_prune' filters
        filters_ordered_by_gradient = np.argsort(-weighted_gradients.detach().cpu().numpy())[:-num_filters_to_prune]
        zeros_mask_dict[param_name].mask = RankedFiltersParameterPruner.mask_from_filter_order(filters_ordered_by_gradient,
                                                                                               param, num_filters)
        msglogger.info("GradientRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
