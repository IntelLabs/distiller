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
from .ranked_structures_pruner import *
from distiller.utils import *
from functools import partial


class AutomatedGradualPrunerBase(_ParameterPruner):
    """Prune to an exact sparsity level specification using a prescribed sparsity
    level schedule formula.

    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    (https://arxiv.org/pdf/1710.01878.pdf)
    """

    def __init__(self, name, initial_sparsity, final_sparsity):
        super().__init__(name)
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        assert final_sparsity > initial_sparsity

    def compute_target_sparsity(self, meta):
        starting_epoch = meta['starting_epoch']
        current_epoch = meta['current_epoch']
        ending_epoch = meta['ending_epoch']
        freq = meta['frequency']
        span = ((ending_epoch - starting_epoch - 1) // freq) * freq
        assert span > 0

        target_sparsity = (self.final_sparsity +
                           (self.initial_sparsity-self.final_sparsity) *
                           (1.0 - ((current_epoch-starting_epoch)/span))**3)

        return target_sparsity

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        target_sparsity = self.compute_target_sparsity(meta)
        self.prune_to_target_sparsity(param, param_name, zeros_mask_dict, target_sparsity, meta['model'])

    def prune_to_target_sparsity(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
        raise NotImplementedError


class AutomatedGradualPruner(AutomatedGradualPrunerBase):
    """Fine-grained pruning with an AGP sparsity schedule.

    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    """
    def __init__(self, name, initial_sparsity, final_sparsity, weights):
        super().__init__(name, initial_sparsity, final_sparsity)
        self.params_names = weights
        assert self.params_names

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.params_names:
            return
        super().set_param_mask(param, param_name, zeros_mask_dict, meta)

    def prune_to_target_sparsity(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
        return SparsityLevelParameterPruner.prune_level(param, param_name, zeros_mask_dict, target_sparsity)


class StructuredAGP(AutomatedGradualPrunerBase):
    """Structured pruning with an AGP sparsity schedule.

    This is a base-class for structured pruning with an AGP schedule.  It is an
    extension of the AGP concept introduced by Zhu et. al.
    """
    def __init__(self, name, initial_sparsity, final_sparsity):
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = None

    def prune_to_target_sparsity(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        self.pruner.prune_to_target_sparsity(param, param_name, zeros_mask_dict, target_sparsity, model)


# TODO: this class parameterization is cumbersome: the ranking functions (per structure)
# should come from the YAML schedule
class L1RankedStructureParameterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = L1RankedStructureParameterPruner(name, group_type, desired_sparsity=0,
                                                       weights=weights, group_dependency=group_dependency)


class ActivationAPoZRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = ActivationAPoZRankedFilterPruner(name, group_type, desired_sparsity=0,
                                                       weights=weights, group_dependency=group_dependency)


class GradientRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = GradientRankedFilterPruner(name, group_type, desired_sparsity=0,
                                                 weights=weights, group_dependency=group_dependency)


class RandomRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = RandomRankedFilterPruner(name, group_type, desired_sparsity=0,
                                               weights=weights, group_dependency=group_dependency)
