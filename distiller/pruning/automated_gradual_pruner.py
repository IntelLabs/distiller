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
# import logging
# msglogger = logging.getLogger()


class AutomatedGradualPruner(_ParameterPruner):
    """Prune to an exact pruning level specification.

    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    (https://arxiv.org/pdf/1710.01878.pdf)
    """

    def __init__(self, name, initial_sparsity, final_sparsity, weights,
                 pruning_fn=None):
        super(AutomatedGradualPruner, self).__init__(name)
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        assert final_sparsity > initial_sparsity
        self.params_names = weights
        assert self.params_names
        if pruning_fn is None:
            self.pruning_fn = self.prune_to_target_sparsity
        else:
            self.pruning_fn = pruning_fn

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.params_names:
            return

        starting_epoch = meta['starting_epoch']
        current_epoch = meta['current_epoch']
        ending_epoch = meta['ending_epoch']
        freq = meta['frequency']
        span = ((ending_epoch - starting_epoch - 1) // freq) * freq
        assert span > 0

        target_sparsity = (self.final_sparsity +
                           (self.initial_sparsity-self.final_sparsity) *
                           (1.0 - ((current_epoch-starting_epoch)/span))**3)
        self.pruning_fn(param, param_name, zeros_mask_dict, target_sparsity, meta['model'])

    @staticmethod
    def prune_to_target_sparsity(param, param_name, zeros_mask_dict, target_sparsity, model=None):
        return SparsityLevelParameterPruner.prune_level(param, param_name, zeros_mask_dict, target_sparsity)


class CriterionParameterizedAGP(AutomatedGradualPruner):
    def __init__(self, name, initial_sparsity, final_sparsity, reg_regims):
        self.reg_regims = reg_regims
        weights = [weight for weight in reg_regims.keys()]
        if not all([group in ['3D', 'Filters', 'Channels', 'Rows'] for group in reg_regims.values()]):
            raise ValueError("Unsupported group structure")
        super(CriterionParameterizedAGP, self).__init__(name, initial_sparsity,
                                                        final_sparsity, weights,
                                                        pruning_fn=self.prune_to_target_sparsity)

    def prune_to_target_sparsity(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        if self.reg_regims[param_name] in ['3D', 'Filters']:
            self.filters_ranking_fn(target_sparsity, param, param_name, zeros_mask_dict, model)
        elif self.reg_regims[param_name] == 'Channels':
            self.channels_ranking_fn(target_sparsity, param, param_name, zeros_mask_dict, model)
        elif self.reg_regims[param_name] == 'Rows':
            self.rows_ranking_fn(target_sparsity, param, param_name, zeros_mask_dict, model)


# TODO: this class parameterization is cumbersome: the ranking functions (per structure)
# should come from the YAML schedule

class L1RankedStructureParameterPruner_AGP(CriterionParameterizedAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, reg_regims):
        super(L1RankedStructureParameterPruner_AGP, self).__init__(name, initial_sparsity, final_sparsity, reg_regims)
        self.filters_ranking_fn = L1RankedStructureParameterPruner.rank_prune_filters
        self.channels_ranking_fn = L1RankedStructureParameterPruner.rank_prune_channels
        self.rows_ranking_fn = L1RankedStructureParameterPruner.rank_prune_rows


class ActivationAPoZRankedFilterPruner_AGP(CriterionParameterizedAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, reg_regims):
        super(ActivationAPoZRankedFilterPruner_AGP, self).__init__(name, initial_sparsity, final_sparsity, reg_regims)
        self.filters_ranking_fn = ActivationAPoZRankedFilterPruner.rank_prune_filters


class GradientRankedFilterPruner_AGP(CriterionParameterizedAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, reg_regims):
        super(GradientRankedFilterPruner_AGP, self).__init__(name, initial_sparsity, final_sparsity, reg_regims)
        self.filters_ranking_fn = GradientRankedFilterPruner.rank_prune_filters


class RandomRankedFilterPruner_AGP(CriterionParameterizedAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, reg_regims):
        super(RandomRankedFilterPruner_AGP, self).__init__(name, initial_sparsity, final_sparsity, reg_regims)
        self.filters_ranking_fn = RandomRankedFilterPruner.rank_prune_filters
