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

from contextlib import contextmanager
import torch
from torchnet.meter import AverageValueMeter
import logging
msglogger = logging.getLogger()

__all__ = ['ActivationStatsCollector', 'collector_context', 'collectors_context']


class ActivationStatsCollector(object):
    """Collect model activation statistics information.

    CNN models with ReLU layers, exhibit sparse activations.
    ActivationStatsCollector will collect information about this sparsity and possibly
    other activations statistics.
    You may collect statistics on different phases of the optimization process (training
    and/or validation).  Currently, we average the values that we collect accross the
    various activations.

    The current implementation has a few caveats:
    * It is slow - therefore it is advisable to use this only when needed.
    * It can't access the activations of torch.Functions, only torch.Modules.
    * The layer names are mangled

    ActivationStatsCollector uses the forward hook of modules in order to access the
    feature-maps.  This is both slow and limits us to seeing only the outputs of torch.Modules.
    We can remove some of the slowness, by choosing to log only specific layers or use it only
    during validation.  By default, we only logs torch.nn.ReLU activations.

    The layer names are mangled, because torch.Modules don't have names and we need to invent
    a unique name per layer.
    """
    def __init__(self, model, statistics_dict, classes=[torch.nn.ReLU]):
        """
        Args:
            model - the model we are monitoring.
            statistics_dict - a dictionary of {stat_name: statistics_function}, where name
                provides a means for us to access the statistics data at a later time; and the
                statistics_function is a function that gets an activation as input and returns
                some statistic.
                For example, the dictionary below collects element-wise activation sparsity
                statistics:
                    {"sparsity": distiller.utils.sparsity}
            classes - a list of class types for which we collect activation statistics.
                You can access a module's activation statistics by referring to module.<stat_name>
                For example:
                    print(module.sparsity)
        """

        super(ActivationStatsCollector, self).__init__()
        self.model = model
        self.statistics_dict = statistics_dict
        self.classes = classes
        self.hook_handles = []

    def value(self):
        """Return a dictionary containing {layer_name: mean statistic}"""
        activation_stats = {}
        self.__collect_activations_stats(self.model, activation_stats)
        return activation_stats

    def __start(self, module, name=''):
        def __activation_stats_cb(module, input, output):
            """Record the activation sparsity of 'module'

            This is a callback from the forward() of 'module'.
            """
            for stat_name, stat_fn in self.statistics_dict.items():
                getattr(module, stat_name).add(stat_fn(output.data))

        is_leaf_node = True
        for name, sub_module in module._modules.items():
            self.__start(sub_module, name)
            is_leaf_node = False

        if is_leaf_node:
            if type(module) in self.classes:
                self.hook_handles.append(module.register_forward_hook(__activation_stats_cb))
                for stat_name, stat_fn in self.statistics_dict.items():
                    if not hasattr(module, stat_name):
                        setattr(module, stat_name, AverageValueMeter())
                        if hasattr(module, 'ref_name'):
                            getattr(module, stat_name).name = stat_name + '_' + module.ref_name
                        else:
                            getattr(module, stat_name).name = stat_name + '_' + name + '_' + \
                                                   module.__class__.__name__ + '_' + str(id(module))

    def start(self):
        """Start collecting activation stats.

        This will iteratively register for the modules' forward-hooks.
        """
        assert len(self.hook_handles) == 0
        self.__start(self.model)

    def stop(self):
        """Stop collecting activation stats.

        This will iteratively unregister for the modules' forward-hooks.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def reset(self):
        """Reset the statsitics counter"""
        self.__reset(self.model)
        return self

    def __reset(self, mod):
        for name, module in mod._modules.items():
            self.__reset(module)
        for stat_name, stat_fn in self.statistics_dict.items():
            if hasattr(mod, stat_name):
                getattr(mod, stat_name).reset()

    def __collect_activations_stats(self, model, activation_stats, name=''):
        for name, module in model._modules.items():
            self.__collect_activations_stats(module, activation_stats, name)

        for stat_name, stat_fn in self.statistics_dict.items():
            if hasattr(model, stat_name):
                activation_stats[getattr(model, stat_name).name] = getattr(model, stat_name).mean


@contextmanager
def collector_context(collector):
    """A context manager for an activation collector"""
    if collector is not None:
        collector.reset().start()
    yield collector
    if collector is not None:
        collector.stop()


@contextmanager
def collectors_context(collectors_dict):
    """A context manager for a dictionary of collectors"""
    if len(collectors_dict) == 0:
        yield None
        return
    for collector in collectors_dict.values():
        collector.reset().start()
    yield None
    for collector in collectors_dict.values():
        collector.stop()


class TrainingProgressCollector(object):
    def __init__(self, stats={}):
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
