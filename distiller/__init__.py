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

from .utils import *
from .thresholding import GroupThresholdMixin, threshold_mask
from .config import file_config, dict_config
from .model_summaries import *
from .scheduler import *
from .sensitivity import *
from .directives import *
from .policy import *
from .thinning import *
from .knowledge_distillation import KnowledgeDistillationPolicy, DistillationLossWeights

#del utils
del dict_config
del thinning
#del model_summaries
#del scheduler
#del sensitivity
#del directives
#del thresholding
#del policy

# Distiller version
__version__ = "0.3.0-pre"

def model_find_param_name(model, param_to_find):
    """Look up the name of a model parameter.

    Arguments:
        model: the model to search
        param_to_find: the parameter whose name we want to look up

    Returns:
        The parameter name (string) or None, if the parameter was not found.
    """
    for name, param  in model.named_parameters():
        if param is param_to_find:
            return name
    return None


def model_find_module_name(model, module_to_find):
    """Look up the name of a module in a model.

    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up

    Returns:
        The module name (string) or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if m == module_to_find:
            return name
    return None

def model_find_param(model, param_to_find_name):
    """Look a model parameter by its name

    Arguments:
        model: the model to search
        param_to_find_name: the name of the parameter that we are searching for

    Returns:
        The parameter or None, if the paramter name was not found.
    """
    for name, param in model.named_parameters():
        if name == param_to_find_name:
            return param
    return None


def model_find_module(model, module_to_find):
    """Given a module name, find the module in the provided model.

    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up

    Returns:
        The module or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if name == module_to_find:
            return m
    return None
