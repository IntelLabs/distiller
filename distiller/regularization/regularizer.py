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

EPSILON = 1e-8

class _Regularizer(object):
    def __init__(self, name, model, reg_regims, threshold_criteria):
        """Regularization base class.

        Args:
            reg_regims: regularization regiment.  A dictionary of
                        reg_regims[<param-name>] = [ lambda, structure-type]
        """
        self.name = name
        self.model = model
        self.reg_regims = reg_regims
        self.threshold_criteria = threshold_criteria

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        raise NotImplementedError

    def threshold(self, param, param_name, zeros_mask_dict):
        raise NotImplementedError
