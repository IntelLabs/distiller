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
import math
import logging
import torch
from torch.nn import functional as f
from random import uniform
import distiller
from .pruner import _ParameterPruner


__all__ = ["LpRankedStructureParameterPruner",
           "L1RankedStructureParameterPruner",
           "L2RankedStructureParameterPruner",
           "RandomLevelStructureParameterPruner",
           "ActivationRankedFilterPruner",
           "ActivationAPoZRankedFilterPruner",
           "ActivationMeanRankedFilterPruner",
           "RandomRankedFilterPruner",
           "BernoulliFilterPruner",
           "GradientRankedFilterPruner",
           "FMReconstructionChannelPruner"]
msglogger = logging.getLogger(__name__)


class _RankedStructureParameterPruner(_ParameterPruner):
    """Base class for pruning structures by ranking them.
    """
    def __init__(self, name, group_type, desired_sparsity, weights, 
                 group_dependency=None, group_size=1, rounding_fn=math.floor, noise=0.):
        super().__init__(name)
        self.group_type = group_type
        self.group_dependency = group_dependency
        self.params_names = weights
        assert self.params_names
        self.leader_binary_map = None
        self.last_target_sparsity = None
        self.desired_sparsity = desired_sparsity
        self.group_size = group_size
        self.rounding_fn = rounding_fn
        self.noise = noise

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
        return self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, fraction_to_prune, model)

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model):
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


class LpRankedStructureParameterPruner(_RankedStructureParameterPruner):
    """Uses Lp-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning), by
    first ranking (sorting) the structures based on their Lp-norm, and then pruning a perctenage
    of the lower-ranking structures.
    See also: https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None, magnitude_fn=None,
                 noise=0.0, group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, desired_sparsity, weights, 
                         group_dependency, group_size, rounding_fn, noise)
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
        if self.group_type in ('3D', 'Filters'):
            group_pruning_fn = partial(self.rank_and_prune_filters, noise=self.noise)
        elif self.group_type in ('Channels', 'Rows'):
            group_pruning_fn = partial(self.rank_and_prune_channels, noise=self.noise)
        elif self.group_type == 'Blocks':
            group_pruning_fn = partial(self.rank_and_prune_blocks, block_shape=self.block_shape)

        binary_map = group_pruning_fn(fraction_to_prune, param, param_name,
                                      zeros_mask_dict, model, binary_map,
                                      magnitude_fn=self.magnitude_fn,
                                      group_size=self.group_size)
        return binary_map

    @staticmethod
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None, zeros_mask_dict=None, 
                                model=None, binary_map=None, magnitude_fn=distiller.norms.l1_norm,
                                noise=0.0, group_size=1, rounding_fn=math.floor):
        if binary_map is None:
            bottomk_channels, channel_mags = distiller.norms.rank_channels(param, group_size, magnitude_fn,
                                                                           fraction_to_prune, rounding_fn, noise)
            if bottomk_channels is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return
            threshold = bottomk_channels[-1]
            binary_map = channel_mags.gt(threshold).type(param.data.type())

        if zeros_mask_dict is not None:
            mask, _ = distiller.thresholding.expand_binary_map(param, 'Channels', binary_map)
            zeros_mask_dict[param_name].mask = mask
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                           magnitude_fn, param_name,
                           distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                           fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map

    @staticmethod
    def rank_and_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict,
                               model=None, binary_map=None, magnitude_fn=distiller.norms.l1_norm,
                               noise=0.0, group_size=1, rounding_fn=math.floor):
        assert param.dim() == 4 or param.dim() == 3, "This pruning is only supported for 3D and 4D weights"
        if binary_map is None:
            bottomk_filters, filter_mags = distiller.norms.rank_filters(param, group_size, magnitude_fn,
                                                                        fraction_to_prune, rounding_fn, noise)
            if bottomk_filters is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                msglogger.info("Too few filters - can't prune %.1f%% filters", 100 * fraction_to_prune)
                return
            threshold = bottomk_filters[-1]
            binary_map = filter_mags.gt(threshold).type(param.data.type())

        if zeros_mask_dict is not None:
            mask, _ = distiller.thresholding.expand_binary_map(param, 'Filters', binary_map)
            zeros_mask_dict[param_name].mask = mask
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f",
                           magnitude_fn, param_name,
                           distiller.sparsity(mask),
                           fraction_to_prune)
        return binary_map

    @staticmethod
    def rank_and_prune_blocks(fraction_to_prune, param, param_name=None, zeros_mask_dict=None,
                              model=None, binary_map=None, block_shape=None,
                              magnitude_fn=distiller.norms.l1_norm, group_size=1):
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

        num_filters, num_channels = param.size(0), param.size(1)
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

        if zeros_mask_dict is not None:
            zeros_mask_dict[param_name].mask = binary_map_to_mask(binary_map, param)
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                           magnitude_fn, param_name,
                           distiller.sparsity_blocks(zeros_mask_dict[param_name].mask, block_shape=block_shape),
                           fraction_to_prune, binary_map.sum().item(), num_super_blocks)
        return binary_map


class L1RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L1-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None, noise=0.0,
                 group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency, 
                         kwargs, magnitude_fn=distiller.norms.l1_norm, noise=noise,
                         group_size=group_size, rounding_fn=rounding_fn)


class L2RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L2-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """
    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None, noise=0.0,
                 group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency, 
                         kwargs, magnitude_fn=distiller.norms.l2_norm, noise=noise,
                         group_size=group_size, rounding_fn=rounding_fn)


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


def _mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map):
    if binary_map is None:
        binary_map = torch.zeros(num_filters).cuda()
        binary_map[filters_ordered_by_criterion] = 1

    expanded = binary_map.expand(np.prod(list(param.size()[1:])), param.size(0)).t().contiguous()
    return distiller.thresholding.expand_binary_map(param, "Filters", binary_map)


class ActivationRankedFilterPruner(_RankedStructureParameterPruner):
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
            raise ValueError("Could not find attribute \"{}\" in module {}"
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
        mask, binary_map = _mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map)
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


class RandomRankedFilterPruner(_RankedStructureParameterPruner):
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
        mask, binary_map = _mask_from_filter_order(filters_ordered_randomly, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("RandomRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


class BernoulliFilterPruner(_RankedStructureParameterPruner):
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
        mask, _ = _mask_from_filter_order(None, param, num_filters, binary_map)
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


class GradientRankedFilterPruner(_RankedStructureParameterPruner):
    """Taylor expansion ranking.

    Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning Convolutional Neural
    Networks for Resource Efficient Inference. ArXiv, abs/1611.06440, 2016.
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
        mask, binary_map = _mask_from_filter_order(filters_ordered_by_gradient, param, num_filters, binary_map)
        zeros_mask_dict[param_name].mask = mask

        msglogger.info("GradientRankedFilterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                       param_name,
                       distiller.sparsity_3D(zeros_mask_dict[param_name].mask),
                       fraction_to_prune, num_filters_to_prune, num_filters)
        return binary_map


from sklearn.linear_model import LinearRegression

def _least_square_sklearn(X, Y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    return model.coef_


def _param_name_2_layer_name(param_name):
    """Convert a weights tensor's name to the name of the layer using the tensor.
    
    By convention, PyTorch modules name their weights parameters as self.weight
    (see for example: torch.nn.modules.conv) which means that their fully-qualified 
    name when enumerating a model's parameters is the modules name followed by '.weight'.
    We exploit this convention to convert a weights tensor name to the fully-qualified 
    module name."""
    return param_name[:-len('.weight')]


class FMReconstructionChannelPruner(_RankedStructureParameterPruner):
    """Uses feature-map (channel) reconstruction to prune weights tensors.

    The idea behind this pruner is to find a reduced subset of the weights, which best
    reconstructs the output of a given layer.  To choose the subset of the weights,
    we use a provided magnitude function to rank the channels of a weights tensor.
    Removing channels from a Convolution layer's weights, means that the layer's input 
    is also reduced. 
    We aim to estimate the minimum mean squared error (MMSE) of the reconstructed outputs,
    given a size-reduced input. The coefficients of the solution to MMSE are then used as
    the new weights of the Convolution layer.

    You must invoke function collect_intermediate_featuremap_samples() prior to using a
    FMReconstructionChannelPruner.  Pass in your model, forward_fn, and module_filter_fn.
    For the fm_caching_fwd_hook argument of collect_intermediate_featuremap_samples, pass
    FMReconstructionChannelPruner.cache_featuremaps_fwd_hook.

    We thank Prof. Han (https://songhan.mit.edu) and his team for their help with 
    this implementation.

    A variant of this technique was used in [1] and [2].

    [1] Channel Pruning for Accelerating Very Deep Neural Networks.
        Yihui He, Xiangyu Zhang, Jian Sun.
        arXiv:1707.06168
    [2] AMC: AutoML for Model Compression and Acceleration on Mobile Devices.
        Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
        arXiv:1802.03494
    """

    @staticmethod
    def cache_featuremaps_fwd_hook(module, input, output, intermediate_fms, n_points_per_fm):
        """Create a cached dictionary of each layer's input and output feature-maps.

        For reconstruction of weights, we need to collect pairs of (layer_input, layer_output)
        using a sample subset of the input dataset.
        This is a forward-hook function, invoked from forward_hooks of Convolution layers.
        Use this in conjunction with distiller.features_collector.collect_intermediate_featuremap_samples,
        which orchestrates the process of feature-map collection.

        This foward-hook samples random points in the output feature-maps of 'module'.
        After collecting the feature-map samples, distiller.FMReconstructionChannelPruner can be used.

        Arguments:
            module - the module who's forward_hook is invoked
            input, output - the input and output arguments to the forward_hook
            intermediate_fms - a dictionary of lists of feature-map samples, per layer 
                (use module.distiller_name as key)
            n_points_per_fm - number of points to sample, per feature-map.
        """
        def im2col(x, conv):
            x_unfold = f.unfold(x, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding)
            return x_unfold

        # Sample random (uniform) points in each feature-map.
        # This method is biased toward small feature-maps.
        if isinstance(module, torch.nn.Conv2d):
            randx = np.random.randint(0, output.size(2), n_points_per_fm)
            randy = np.random.randint(0, output.size(3), n_points_per_fm)

        X = input[0]
        if isinstance(module, torch.nn.Linear):
            X = X.detach().cpu().clone()
            Y = output.detach().cpu().clone()
        elif module.kernel_size == (1, 1):
            X = X[:, :, randx, randy].detach().cpu().clone()
            Y = output[:, :, randx, randy].detach().cpu().clone()
        else:
            w, h = X.size(2), X.size(3)
            X = im2col(X.detach().cpu().clone(), module).squeeze()
            w_out = output.size(2)
            pts = randx * w_out + randy
            X = X[:, :, pts].detach().cpu().clone()
            Y = output[:, :, randx, randy].detach().cpu().clone()

        # Preprocess the outputs: transpose the batch and channel dimensions, create a flattened view, and transpose.
        # The outputs originally have shape: (batch size, num channels, feature-map width, feature-map height).
        Y = Y.view(Y.size(0), Y.size(1), -1)
        Y = Y.transpose(2, 1)
        Y = Y.contiguous().view(-1, Y.size(2))

        intermediate_fms['output_fms'][module.distiller_name].append(Y)
        intermediate_fms['input_fms'][module.distiller_name].append(X)

    def __init__(self, name, group_type, desired_sparsity, weights,
                 group_dependency=None, kwargs=None, magnitude_fn=distiller.norms.l1_norm,
                 group_size=1, rounding_fn=math.floor, ranking_noise=0.):
        super().__init__(name, group_type, desired_sparsity, weights, group_dependency,
                         group_size=group_size, rounding_fn=rounding_fn, noise=ranking_noise)
        if group_type != "Channels":
            raise ValueError("Structure {} was requested but "
                             "currently ranking of this shape is not supported".
                             format(group_type))
        assert magnitude_fn is not None
        self.magnitude_fn = magnitude_fn

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return

        binary_map = self.rank_and_prune_channels(fraction_to_prune, param, param_name,
                                                  zeros_mask_dict, model, binary_map,
                                                  group_size=self.group_size,
                                                  rounding_fn=self.rounding_fn,
                                                  noise=self.noise)
        return binary_map

    @staticmethod
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None,
                                zeros_mask_dict=None, model=None, binary_map=None, 
                                magnitude_fn=distiller.norms.l1_norm, group_size=1, rounding_fn=math.floor,
                                noise=0):
        assert binary_map is None
        if binary_map is None:
            bottomk_channels, channel_mags = distiller.norms.rank_channels(param, group_size, magnitude_fn,
                                                                           fraction_to_prune, rounding_fn, noise)

            # Todo: this little piece of code can be refactored
            if bottomk_channels is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return

            threshold = bottomk_channels[-1]
            binary_map = channel_mags.gt(threshold)

            # These are the indices of channels we want to keep
            indices = binary_map.nonzero().squeeze()
            if len(indices.shape) == 0:
                indices = indices.expand(1)

            # Find the module representing this layer
            distiller.assign_layer_fq_names(model)
            layer_name = _param_name_2_layer_name(param_name)
            conv = distiller.find_module_by_fq_name(model, layer_name)
            try:
                Y = model.intermediate_fms['output_fms'][layer_name]
                X = model.intermediate_fms['input_fms'][layer_name]
            except AttributeError:
                raise ValueError("To use FMReconstructionChannelPruner you must first collect input statistics")

            op_type = 'conv' if param.dim() == 4 else 'fc'
            # We need to remove the chosen weights channels.  Because we are using 
            # min(MSE) to compute the weights, we need to start by removing feature-map 
            # channels from the input.  Then we perform the MSE regression to generate
            # a smaller weights tensor.
            if op_type == 'fc':
                X = X[:, binary_map]
            elif conv.kernel_size == (1, 1):
                X = X[:, binary_map, :]
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))
            else:
                # X is (batch, ck^2, num_pts)
                # we want:   (batch, c, k^2, num_pts)
                X = X.view(X.size(0), -1, np.prod(conv.kernel_size), X.size(2))
                X = X[:, binary_map, :, :]
                X = X.view(X.size(0), -1, X.size(3))
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))

            # Approximate the weights given input-FMs and output-FMs
            new_w = _least_square_sklearn(X, Y)
            new_w = torch.from_numpy(new_w) # shape: (num_filters, num_non_masked_channels * k^2)
            cnt_retained_channels = binary_map.sum()

            if op_type == 'conv':
                # Expand the weights back to their original size,
                new_w = new_w.contiguous().view(param.size(0), cnt_retained_channels, param.size(2), param.size(3))

                # Copy the weights that we learned from minimizing the feature-maps least squares error,
                # to our actual weights tensor.
                param.detach()[:, indices, :,   :] = new_w.type(param.type())
            else:
                param.detach()[:, indices] = new_w.type(param.type())

        if zeros_mask_dict is not None:
            binary_map = binary_map.type(param.type())
            if op_type == 'conv':
                zeros_mask_dict[param_name].mask, _ = distiller.thresholding.expand_binary_map(param,
                                                                                               'Channels', binary_map)
                msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                               param_name,
                               distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                               fraction_to_prune, binary_map.sum().item(), param.size(1))
            else:
                msglogger.error("fc sparsity = %.2f" % (1 - binary_map.sum().item() / binary_map.size(0)))
                zeros_mask_dict[param_name].mask = binary_map.expand(param.size(0), param.size(1))
                msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                               param_name,
                               distiller.sparsity_cols(zeros_mask_dict[param_name].mask),
                               fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map
