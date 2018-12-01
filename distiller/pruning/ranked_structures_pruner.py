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


class RankedStructureParameterPruner(_ParameterPruner):
    """Base class for pruning structures by ranking them.
    """
    def __init__(self, name, group_type, desired_sparsity,  weights, group_dependency=None):
        super().__init__(name)
        self.group_type = group_type
        self.group_dependency = group_dependency
        self.params_names = weights
        assert self.params_names
        self.leader_binary_map = None
        self.last_target_sparsity = None
        self.desired_sparsity = desired_sparsity

    def leader(self):
        # The "leader" is the first weights-tensor in the list
        return self.params_names[0]

    def is_supported(self, param_name):
        return param_name in self.params_names

    def fraction_to_prune(self, param_name):
        return self.desired_sparsity

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if not self.is_supported(param_name):
            return
        fraction_to_prune = self.fraction_to_prune(param_name)
        try:
            model = meta['model']
        except TypeError:
            model = None
        return self.prune_to_target_sparsity(param, param_name, zeros_mask_dict, fraction_to_prune, model)

    def prune_to_target_sparsity(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        if not self.is_supported(param_name):
            return

        binary_map = None
        if self.group_dependency == "Leader":
            if target_sparsity != self.last_target_sparsity:
                # Each time we change the target sparsity we need to compute and cache the leader's binary-map.
                # We don't have control over the order that this function is invoked, so the only indication that
                # we need to compute a new leader binary-map is the change of the target_sparsity.
                self.last_target_sparsity = target_sparsity
                self.leader_binary_map = self.prune_group(target_sparsity, model.state_dict()[self.leader()],
                                                          self.leader(), zeros_mask_dict=None)
            assert self.leader_binary_map is not None
            binary_map = self.leader_binary_map
        # Delegate the actual pruning to a sub-class
        self.prune_group(target_sparsity, param, param_name, zeros_mask_dict, model, binary_map)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        raise NotImplementedError


class L1RankedStructureParameterPruner(RankedStructureParameterPruner):
    """Uses mean L1-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)
        if group_type not in ['3D', 'Filters', 'Channels', 'Rows']:
            raise ValueError("Structure {} was requested but"
                             "currently only filter (3D) and channel ranking is supported".
                             format(group_type))

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        if self.group_type in ['3D', 'Filters']:
            group_pruning_fn = self.rank_and_prune_filters
        elif self.group_type == 'Channels':
            group_pruning_fn = self.rank_and_prune_channels
        elif self.group_type == 'Rows':
            group_pruning_fn = self.rank_and_prune_rows

        binary_map = group_pruning_fn(fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map)
        return binary_map

    @staticmethod
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None,
                                zeros_mask_dict=None, model=None, binary_map=None):
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

        def binary_map_to_mask(binary_map, param):
            num_filters = param.size(0)
            num_channels = param.size(1)
            a = binary_map.expand(num_filters, num_channels)
            c = a.unsqueeze(-1)
            d = c.expand(num_filters, num_channels, param.size(2) * param.size(3)).contiguous()
            return d.view(num_filters, num_channels, param.size(2), param.size(3))

        if binary_map is None:
            bottomk_channels, channel_mags = rank_channels(fraction_to_prune, param)
            if bottomk_channels is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return
            threshold = bottomk_channels[-1]
            binary_map = channel_mags.gt(threshold).type(param.data.type())

        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = binary_map_to_mask(binary_map, param)
            msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)", param_name,
                           distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                           fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map

    @staticmethod
    def rank_and_prune_filters(fraction_to_prune, param, param_name,
                               zeros_mask_dict, model=None, binary_map=None):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"

        threshold = None
        if binary_map is None:
            # First we rank the filters
            view_filters = param.view(param.size(0), -1)
            filter_mags = view_filters.data.abs().mean(dim=1)
            topk_filters = int(fraction_to_prune * filter_mags.size(0))
            if topk_filters == 0:
                msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
                return
            bottomk, _ = torch.topk(filter_mags, topk_filters, largest=False, sorted=True)
            threshold = bottomk[-1]
            msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=(%d/%d)",
                           param_name,
                           topk_filters, filter_mags.size(0))
        # Then we threshold
        mask, binary_map = distiller.group_threshold_mask(param, 'Filters', threshold, 'Mean_Abs', binary_map)
        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = mask
        msglogger.info("L1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f",
                       param_name,
                       distiller.sparsity(mask),
                       fraction_to_prune)
        return binary_map

    @staticmethod
    def rank_and_prune_rows(fraction_to_prune, param, param_name,
                            zeros_mask_dict, model=None, binary_map=None):
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


def mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map):
    if binary_map is None:
        binary_map = torch.zeros(num_filters).cuda()
        binary_map[filters_ordered_by_criterion] = 1
    expanded = binary_map.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
    return expanded.view(param.shape), binary_map


class ActivationAPoZRankedFilterPruner(RankedStructureParameterPruner):
    """Uses mean APoZ (average percentage of zeros) activation channels to rank structures
    and prune a specified percentage of structures.

    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures",
    Hengyuan Hu, Rui Peng, Yu-Wing Tai, Chi-Keung Tang, ICLR 2016
    https://arxiv.org/abs/1607.03250
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
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

        # Sort from low to high, and remove the bottom 'num_filters_to_prune' filters
        filters_ordered_by_apoz = np.argsort(apoz)[:-num_filters_to_prune]
        mask, binary_map = mask_from_filter_order(filters_ordered_by_apoz, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("ActivationL1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


class RandomRankedFilterPruner(RankedStructureParameterPruner):
    """A Random raanking of filters.

    This is used for sanity testing of other algorithms.
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)

        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        filters_ordered_randomly = np.random.permutation(num_filters)[:-num_filters_to_prune]
        mask, binary_map = mask_from_filter_order(filters_ordered_randomly, param, num_filters)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("RandomRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


class GradientRankedFilterPruner(RankedStructureParameterPruner):
    """
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
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
        mask, binary_map = mask_from_filter_order(filters_ordered_by_gradient, param, num_filters)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("GradientRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map
