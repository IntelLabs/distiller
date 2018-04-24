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
from .pruner import _ParameterPruner
import distiller
msglogger = logging.getLogger()

class StructureParameterPruner(distiller.GroupThresholdMixin, _ParameterPruner):
    """Prune parameter structures.

    Pruning criterion: average L1-norm.  If the average L1-norm (absolute value) of the eleements
    in the structure is below threshold, then the structure is pruned.

    We use the average, instead of plain L1-norm, because we don't want the threshold to depend on
    the structure size.
    """
    def __init__(self, name, model, reg_regims, threshold_criteria):
        super(StructureParameterPruner, self).__init__(name)
        self.name = name
        self.model = model
        self.reg_regims = reg_regims
        self.threshold_criteria = threshold_criteria
        assert threshold_criteria in ["Max", "Mean_Abs"]

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.reg_regims.keys():
            return

        group_type = self.reg_regims[param_name][1]
        threshold = self.reg_regims[param_name][0]
        zeros_mask_dict[param_name].mask = self.group_threshold_mask(param,
                                                                     group_type,
                                                                     threshold,
                                                                     self.threshold_criteria)
