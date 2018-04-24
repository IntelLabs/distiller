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
from distiller.utils import sparsity
from torchnet.meter import AverageValueMeter
import logging
msglogger = logging.getLogger()

__all__ = ['ActivationSparsityCollector']

class DataCollector(object):
    def __init__(self):
        pass


class ActivationSparsityCollector(DataCollector):
    """Collect model activation sparsity information.

    CNN models with ReLU layers, exhibit sparse activations.
    ActivationSparsityCollector will collect information about this sparsity.
    Currently we only record the mean sparsity of the activations, but this can be expanded
    to collect std and other statistics.

    The current implementation activation sparsity collection has a few caveats:
    * It is slow
    * It can't access the activations of torch.Functions, only torch.Modules.
    * The layer names are mangled

    ActivationSparsityCollector uses the forward hook of modules in order to access the
    feature-maps.  This is both slow and limits us to seeing only the outputs of torch.Modules.
    We can remove some of the slowness, by choosing to log only specific layers.  By default,
    we only logs torch.nn.ReLU activations.

    The layer names are mangled, because torch.Modules don't have names and we need to invent
    a unique name per layer.
    """
    def __init__(self, model, classes=[torch.nn.ReLU]):
        """Since only specific layers produce sparse feature-maps, the
        ActivationSparsityCollector constructor accepts an optional list of layers to log."""

        super(ActivationSparsityCollector, self).__init__()
        self.model = model
        self.classes = classes
        self._init_activations_sparsity(model)

    def value(self):
        """Return a dictionary containing {layer_name: mean sparsity}"""
        activation_sparsity = {}
        _collect_activations_sparsity(self.model, activation_sparsity)
        return activation_sparsity


    def _init_activations_sparsity(self, module, name=''):
        def __activation_sparsity_cb(module, input, output):
            """Record the activation sparsity of 'module'

            This is a callback from the forward() of 'module'.
            """
            module.sparsity.add(sparsity(output.data))

        has_children = False
        for name, sub_module in module._modules.items():
            self._init_activations_sparsity(sub_module, name)
            has_children = True
        if not has_children:
            if type(module) in self.classes:
                module.register_forward_hook(__activation_sparsity_cb)
                module.sparsity = AverageValueMeter()
                if hasattr(module, 'ref_name'):
                    module.sparsity.name = 'sparsity_' + module.ref_name
                else:
                    module.sparsity.name = 'sparsity_' + name + '_' + module.__class__.__name__ + '_' + str(id(module))

    @staticmethod
    def _collect_activations_sparsity(model, activation_sparsity, name=''):
        for name, module in model._modules.items():
            _collect_activations_sparsity(module, activation_sparsity, name)

        if hasattr(model, 'sparsity'):
            activation_sparsity[model.sparsity.name] = model.sparsity.mean


class TrainingProgressCollector(DataCollector):
    def __init__(self, stats = {}):
        super(TrainingProgressCollector, self).__init__()
        object.__setattr__(self, '_stats', stats)

    def __setattr__(self, name, value):
        stats = self.__dict__.get('_stats')
        stats[name] = value

    def __getattr__(self, name):
        if name in self.__dict__['_stats']:
            return self.__dict__['_stats'][name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def value(self):
        return self._stats
