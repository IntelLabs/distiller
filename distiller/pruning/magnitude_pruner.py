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
import distiller

class MagnitudeParameterPruner(_ParameterPruner):
    """This is the most basic magnitude-based pruner.

    This pruner supports configuring a scalar threshold for each layer.
    A default threshold is mandatory and is used for layers without explicit
    threshold setting.

    """
    def __init__(self, name, thresholds, **kwargs):
        super(MagnitudeParameterPruner, self).__init__(name)
        assert 'thresholds' is not None
        # Make sure there is a default threshold to use
        assert '*' in thresholds
        self.thresholds = thresholds

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        threshold = self.thresholds.get(param_name, self.thresholds['*'])
        zeros_mask_dict[param_name].mask = distiller.threshold_mask(param.data, threshold)
