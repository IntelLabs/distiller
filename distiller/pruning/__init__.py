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

"""
:mod:`distiller.pruning` is a package implementing various pruning algorithms.
"""

from .magnitude_pruner import MagnitudeParameterPruner
from .automated_gradual_pruner import AutomatedGradualPruner, L1RankedStructureParameterPruner_AGP, \
                                      ActivationAPoZRankedFilterPruner_AGP, GradientRankedFilterPruner_AGP, \
                                      RandomRankedFilterPruner_AGP
from .level_pruner import SparsityLevelParameterPruner
from .sensitivity_pruner import SensitivityPruner
from .splicing_pruner import SplicingPruner
from .structure_pruner import StructureParameterPruner
from .ranked_structures_pruner import L1RankedStructureParameterPruner, ActivationAPoZRankedFilterPruner, \
                                      RandomRankedFilterPruner, GradientRankedFilterPruner
from .baidu_rnn_pruner import BaiduRNNPruner

del magnitude_pruner
del automated_gradual_pruner
del level_pruner
del sensitivity_pruner
del structure_pruner
del ranked_structures_pruner
