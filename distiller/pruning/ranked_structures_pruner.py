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

from functools import partial
import numpy as np
import logging
import torch
from random import uniform
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


l1_magnitude = partial(torch.norm, p=1)
l2_magnitude = partial(torch.norm, p=2)


class LpRankedStructureParameterPruner(RankedStructureParameterPruner):
    """Uses Lp-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning), by
    first ranking (sorting) the structures based on their Lp-norm, and then pruning a perctenage
    of the lower-ranking structures.
    See also: https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None, magnitude_fn=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)
        if group_type not in ['3D', 'Filters', 'Channels', 'Rows', 'Blocks']:
            raise ValueError("Structure {} was requested but "
                             "currently ranking of this shape is not supported".
                             format(group_type))
        assert magnitude_fn is not None
        self.magnitude_fn = magnitude_fn

        if group_type == 'Blocks':
            try:
                self.block_shape = kwargs['block_shape']
            except KeyError:
                raise ValueError("When defining a block pruner you must also specify the block shape")

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        if self.group_type in ['3D', 'Filters']:
            group_pruning_fn = self.rank_and_prune_filters
        elif self.group_type == 'Channels':
            group_pruning_fn = partial(self.rank_and_prune_channels)
        elif self.group_type == 'Rows':
            group_pruning_fn = self.rank_and_prune_rows
        elif self.group_type == 'Blocks':
            group_pruning_fn = partial(self.rank_and_prune_blocks, block_shape=self.block_shape)

        binary_map = group_pruning_fn(fraction_to_prune, param, param_name,
                                      zeros_mask_dict, model, binary_map,
                                      magnitude_fn=self.magnitude_fn)
        return binary_map

    @staticmethod
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None,
                                zeros_mask_dict=None, model=None, binary_map=None, magnitude_fn=l1_magnitude):
        def rank_channels(fraction_to_prune, param):
            num_filters = param.size(0)
            num_channels = param.size(1)
            kernel_size = param.size(2) * param.size(3)

            # First, reshape the weights tensor such that each channel (kernel) in the original
            # tensor, is now a row in the 2D tensor.
            view_2d = param.view(-1, kernel_size)
            # Next, compute the sums of each kernel
            kernel_mags = magnitude_fn(view_2d, dim=1)
            # Now group by channels
            k_sums_mat = kernel_mags.view(num_filters, num_channels).t()
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

        threshold_type = 'L1' if magnitude_fn == l1_magnitude else 'L2'
        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = binary_map_to_mask(binary_map, param)
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                           threshold_type, param_name,
                           distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                           fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map

    @staticmethod
    def rank_and_prune_filters(fraction_to_prune, param, param_name,
                               zeros_mask_dict, model=None, binary_map=None, magnitude_fn=l1_magnitude):
        assert param.dim() == 4, "This pruning is only supported for 4D weights"

        threshold = None
        threshold_type = 'L1' if magnitude_fn == l1_magnitude else 'L2'
        if binary_map is None:
            # First we rank the filters
            view_filters = param.view(param.size(0), -1)
            filter_mags = magnitude_fn(view_filters, dim=1)
            topk_filters = int(fraction_to_prune * filter_mags.size(0))
            if topk_filters == 0:
                msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
                return
            bottomk, _ = torch.topk(filter_mags, topk_filters, largest=False, sorted=True)
            threshold = bottomk[-1]
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=(%d/%d)",
                           threshold_type, param_name,
                           topk_filters, filter_mags.size(0))
        # Then we threshold
        mask, binary_map = distiller.group_threshold_mask(param, 'Filters', threshold, threshold_type, binary_map)
        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = mask
        msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f",
                       threshold_type, param_name,
                       distiller.sparsity(mask),
                       fraction_to_prune)

        # Compensate for dropping filters
        #param.data /= float(distiller.sparsity(mask))
        return binary_map

    @staticmethod
    def rank_and_prune_rows(fraction_to_prune, param, param_name,
                            zeros_mask_dict, model=None, binary_map=None, magnitude_fn=l1_magnitude):
        """Prune the rows of a matrix, based on ranked L1-norms of the matrix rows.

        PyTorch stores the weights matrices in a transposed format.  I.e. before performing GEMM, a matrix is
        transposed.  This is counter-intuitive.  To deal with this, we can either transpose the matrix and
        then proceed to compute the masks as usual, or we can treat columns as rows, and rows as columns :-(.
        We choose the latter, because transposing very large matrices can be detrimental to performance.  Note
        that computing mean L1-norm of columns is also not optimal, because consequtive column elements are far
        away from each other in memory, and this means poor use of caches and system memory.
        """

        assert param.dim() == 2, "This pruning is only supported for 2D weights"
        ROWS_DIM = 0
        THRESHOLD_DIM = 'Cols'
        rows_mags = magnitude_fn(param, dim=ROWS_DIM)
        num_rows_to_prune = int(fraction_to_prune * rows_mags.size(0))
        if num_rows_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% rows", 100*fraction_to_prune)
            return
        bottomk_rows, _ = torch.topk(rows_mags, num_rows_to_prune, largest=False, sorted=True)
        threshold = bottomk_rows[-1]
        threshold_type = 'L1' if magnitude_fn == l1_magnitude else 'L2'
        zeros_mask_dict[param_name].mask, binary_map = distiller.group_threshold_mask(param, THRESHOLD_DIM,
                                                                                      threshold, threshold_type)
        msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       threshold_type, param_name,
                       distiller.sparsity(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_rows_to_prune, rows_mags.size(0))
        return binary_map

    @staticmethod
    def rank_and_prune_blocks(fraction_to_prune, param, param_name=None, zeros_mask_dict=None,
                              model=None, binary_map=None, block_shape=None, magnitude_fn=l1_magnitude):
        """Block-wise pruning for 4D tensors.

        The block shape is specified using a tuple: [block_repetitions, block_depth, block_height, block_width].
        The dimension 'block_repetitions' specifies in how many consecutive filters the "basic block"
        (shaped as [block_depth, block_height, block_width]) repeats to produce a (4D) "super block".

        For example:

          block_pruner:
            class: L1RankedStructureParameterPruner_AGP
            initial_sparsity : 0.05
            final_sparsity: 0.70
            group_type: Blocks
            kwargs:
              block_shape: [1,8,1,1]  # [block_repetitions, block_depth, block_height, block_width]

        Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1
        """
        if len(block_shape) != 4:
            raise ValueError("The block shape must be specified as a 4-element tuple")
        block_repetitions, block_depth, block_height, block_width = block_shape
        if not block_width == block_height == 1:
            raise ValueError("Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1")
        super_block_volume = distiller.volume(block_shape)
        num_super_blocks = distiller.volume(param) / super_block_volume
        if distiller.volume(param) % super_block_volume != 0:
            raise ValueError("The super-block size must divide the weight tensor exactly.")

        num_filters = param.size(0)
        num_channels = param.size(1)
        kernel_size = param.size(2) * param.size(3)

        if block_depth > 1:
            view_dims = (num_filters*num_channels//(block_repetitions*block_depth),
                         block_repetitions*block_depth,
                         kernel_size,)
        else:
            view_dims = (num_filters // block_repetitions,
                         block_repetitions,
                         -1,)

        def rank_blocks(fraction_to_prune, param):
            # Create a view where each block is a column
            view1 = param.view(*view_dims)
            # Next, compute the sums of each column (block)
            block_mags = magnitude_fn(view1, dim=1)
            block_mags = block_mags.view(-1)  # flatten
            k = int(fraction_to_prune * block_mags.size(0))
            if k == 0:
                msglogger.info("Too few blocks (%d)- can't prune %.1f%% blocks",
                               block_mags.size(0), 100*fraction_to_prune)
                return None, None

            bottomk, _ = torch.topk(block_mags, k, largest=False, sorted=True)
            return bottomk, block_mags

        def binary_map_to_mask(binary_map, param):
            a = binary_map.view(view_dims[0], view_dims[2])
            c = a.unsqueeze(1)
            d = c.expand(*view_dims).contiguous()
            return d.view(num_filters, num_channels, param.size(2), param.size(3))

        if binary_map is None:
            bottomk_blocks, block_mags = rank_blocks(fraction_to_prune, param)
            if bottomk_blocks is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return
            threshold = bottomk_blocks[-1]
            binary_map = block_mags.gt(threshold).type(param.data.type())

        threshold_type = 'L1' if magnitude_fn == l1_magnitude else 'L2'
        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = binary_map_to_mask(binary_map, param)
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                           threshold_type, param_name,
                           distiller.sparsity_blocks(zeros_mask_dict[param_name].mask, block_shape=block_shape),
                           fraction_to_prune, binary_map.sum().item(), num_super_blocks)
        return binary_map


class L1RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L1-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None):
        super().__init__(name, group_type, desired_sparsity, weights,
                         group_dependency, kwargs, magnitude_fn=l1_magnitude)


class L2RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L2-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None):
        super().__init__(name, group_type, desired_sparsity, weights,
                         group_dependency, kwargs, magnitude_fn=l2_magnitude)


class RandomLevelStructureParameterPruner(L1RankedStructureParameterPruner):
    """Uses mean L1-norm to rank and prune structures, with a random pruning regimen.

    This class sets the pruning level to a random value in the range sparsity_range,
    and chooses which structures to remove using L1-norm ranking.
    The idea is similiar to DropFilter, but instead of randomly choosing filters,
    we randomly choose a sparsity level and then prune according to magnitude.
    """
    def __init__(self, name, group_type, sparsity_range, weights,
                 group_dependency=None, kwargs=None):
        self.sparsity_range = sparsity_range
        super().__init__(name, group_type, 0, weights, group_dependency, kwargs)

    def fraction_to_prune(self, param_name):
        return uniform(self.sparsity_range[0], self.sparsity_range[1])


def mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map):
    if binary_map is None:
        binary_map = torch.zeros(num_filters).cuda()
        binary_map[filters_ordered_by_criterion] = 1
    expanded = binary_map.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
    return expanded.view(param.shape), binary_map


class ActivationRankedFilterPruner(RankedStructureParameterPruner):
    """Base class for pruners ranking convolution filters by some quality criterion of the
    corresponding feature-map channels (e.g. mean channel activation L1 value).
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    @property
    def activation_rank_criterion(self):
        raise NotImplementedError

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name,
                                                 zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This pruning is only supported for 4D weights"

        # Use the parameter name to locate the module that has the activation sparsity statistics
        fq_name = param_name.replace(".conv", ".relu")[:-len(".weight")]
        module = distiller.find_module_by_fq_name(model, fq_name)
        if module is None:
            raise ValueError("Could not find a layer named %s in the model."
                             "\nMake sure to use assign_layer_fq_names()" % fq_name)
        if not hasattr(module, self.activation_rank_criterion):
            raise ValueError("Could not find attribute \"{}\" in module %s"
                             "\nMake sure to use SummaryActivationStatsCollector(\"{}\")".
                             format(self.activation_rank_criterion, fq_name, self.activation_rank_criterion))

        quality_criterion, std = getattr(module, self.activation_rank_criterion).value()
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)
        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        # Sort from low to high, and remove the bottom 'num_filters_to_prune' filters
        filters_ordered_by_criterion = np.argsort(quality_criterion)[:-num_filters_to_prune]
        mask, binary_map = mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("ActivationL1RankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


class ActivationAPoZRankedFilterPruner(ActivationRankedFilterPruner):
    """Uses mean APoZ (average percentage of zeros) activation channels to rank filters
    and prune a specified percentage of filters.

    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures,"
    Hengyuan Hu, Rui Peng, Yu-Wing Tai, Chi-Keung Tang. ICLR 2016.
    https://arxiv.org/abs/1607.03250
    """
    @property
    def activation_rank_criterion(self):
        return 'apoz_channels'


class ActivationMeanRankedFilterPruner(ActivationRankedFilterPruner):
    """Uses mean value of activation channels to rank filters and prune a specified percentage of filters.

    "Pruning Convolutional Neural Networks for Resource Efficient Inference,"
    Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz. ICLR 2017.
    https://arxiv.org/abs/1611.06440
    """
    @property
    def activation_rank_criterion(self):
        return 'mean_channels'


class RandomRankedFilterPruner(RankedStructureParameterPruner):
    """A Random ranking of filters.

    This is used for sanity testing of other algorithms.
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name,
                                                 zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This pruning is only supported for 4D weights"
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)

        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        filters_ordered_randomly = np.random.permutation(num_filters)[:-num_filters_to_prune]
        mask, binary_map = mask_from_filter_order(filters_ordered_randomly, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("RandomRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


class BernoulliFilterPruner(RankedStructureParameterPruner):
    """A Bernoulli probability for dropping each filter.

    This is can be used for random filter-dropping algorithms (e.g. DropFilter)
    """
    def __init__(self, name, group_type, desired_sparsity, weights, group_dependency=None):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name,
                                                 zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This pruner is only supported for 4D weights"
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)

        keep_prob = 1 - fraction_to_prune
        if binary_map is None:
            binary_map = torch.bernoulli(torch.as_tensor([keep_prob] * num_filters))
        mask, _ = mask_from_filter_order(None, param, num_filters, binary_map)
        # mask = mask.detach()
        mask = mask.to(param.device)
        # Compensate for dropping filters
        pruning_factor = binary_map.sum() / num_filters
        mask.div_(pruning_factor)
        zeros_mask_dict[param_name].mask = mask
        msglogger.debug("BernoulliFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
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
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name,
                                                 zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This pruning is only supported for 4D weights"
        if param.grad is None:
            msglogger.info("Skipping gradient pruning of %s because it does not have a gradient yet", param_name)
            return
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)
        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100*fraction_to_prune)
            return

        # Compute the multiplication of the filters times the filter_gradienrs
        view_filters = param.view(param.size(0), -1)
        view_filter_grads = param.grad.view(param.size(0), -1)
        weighted_gradients = view_filter_grads * view_filters
        weighted_gradients = weighted_gradients.sum(dim=1)

        # Sort from high to low, and remove the bottom 'num_filters_to_prune' filters
        filters_ordered_by_gradient = np.argsort(-weighted_gradients.detach().cpu().numpy())[:-num_filters_to_prune]
        mask, binary_map = mask_from_filter_order(filters_ordered_by_gradient, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("GradientRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map
