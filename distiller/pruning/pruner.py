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

import torch
import distiller

class _ParameterPruner(object):
    """Base class for all pruners.

    Arguments:
        name: pruner name is used mainly for debugging.
    """
    def __init__(self, name):
        self.name = name

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        raise NotImplementedError

def threshold_model(model, threshold):
    """Threshold an entire model using the provided threshold

    This function prunes weights only (biases are left untouched).
    """
    for name, p in model.named_parameters():
       if 'weight' in name:
           mask = distiller.threshold_mask(param.data, threshold)
           p.data = p.data.mul_(mask)
