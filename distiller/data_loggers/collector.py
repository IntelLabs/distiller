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

from functools import partial
import xlsxwriter
import os
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torchnet.meter import AverageValueMeter
import logging
import distiller
msglogger = logging.getLogger()

__all__ = ['SummaryActivationStatsCollector', 'RecordsActivationStatsCollector',
           'collector_context', 'collectors_context']


class ActivationStatsCollector(object):
    """Collect model activation statistics information.

    ActivationStatsCollector is the base class for classes that collect activations statistics.
    You may collect statistics on different phases of the optimization process (training, validation, test).

    Statistics data are accessible via .value() or by accessing individual modules.

    The current implementation has a few caveats:
    * It is slow - therefore it is advisable to use this only when needed.
    * It can't access the activations of torch.Functions, only torch.Modules.

    ActivationStatsCollector uses the forward hook of modules in order to access the
    feature-maps.  This is both slow and limits us to seeing only the outputs of torch.Modules.
    We can remove some of the slowness, by choosing to log only specific layers or use it only
    during validation or test.  By default, we only log torch.nn.ReLU activations.

    The layer names are mangled, because torch.Modules don't have names and we need to invent
    a unique name per layer.  To assign human-readable names, it is advisable to invoke the following
    before starting the statistics collection:

        distiller.utils.assign_layer_fq_names(model)
    """
    def __init__(self, model, stat_name, classes):
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
        self.stat_name = stat_name
        self.classes = classes
        self.fwd_hook_handles = []

    def value(self):
        """Return a dictionary containing {layer_name: statistic}"""
        activation_stats = OrderedDict()
        self.model.apply(partial(self._collect_activations_stats, activation_stats=activation_stats))
        return activation_stats

    def start(self):
        """Start collecting activation stats.

        This will iteratively register the modules' forward-hooks, so that the collector
        will be called from the forward traversal and get exposed to activation data.
        """
        assert len(self.fwd_hook_handles) == 0
        self.model.apply(self.start_module)

    def start_module(self, module):
        """Iteratively register to the forward-pass callback of all eligable modules.

        Eligable modules are currently filtered by their class type.
        """
        is_leaf_node = len(list(module.children())) == 0
        if is_leaf_node and type(module) in self.classes:
            self.fwd_hook_handles.append(module.register_forward_hook(self._activation_stats_cb))
            self._start_counter(module)

    def stop(self):
        """Stop collecting activation stats.

        This will iteratively unregister the modules' forward-hooks.
        """
        for handle in self.fwd_hook_handles:
            handle.remove()
        self.fwd_hook_handles = []

    def reset(self):
        """Reset the statsitics counters of this collector."""
        self.model.apply(self._reset_counter)
        return self

    def __activation_stats_cb(self, module, input, output):
        """Handle new activations ('output' argument).

        This is invoked from the forward-pass callback of module 'module'.
        """
        raise NotImplementedError

    def _reset_counter(self, module):
        """Reset a specific statistic counter - this is subclass-specific code"""
        raise NotImplementedError

    def _collect_activations_stats(self, module, activation_stats, name=''):
        """Handle new activations - this is subclass-specific code"""
        raise NotImplementedError


class SummaryActivationStatsCollector(ActivationStatsCollector):
    """This class collects activiations statistical summaries.

    This Collector computes the mean of some statistic of the activation.  It is rather
    light-weight and quicker than collecting a record per activation.
    The statistic function is configured in the constructor.
    """
    def __init__(self, model, stat_name, summary_fn, classes=[torch.nn.ReLU]):
        super(SummaryActivationStatsCollector, self).__init__(model, stat_name, classes)
        self.summary_fn = summary_fn

    def _activation_stats_cb(self, module, input, output):
        """Record the activation sparsity of 'module'

        This is a callback from the forward() of 'module'.
        """
        try:
            getattr(module, self.stat_name).add(self.summary_fn(output.data))
        except RuntimeError as e:
            if "The expanded size of the tensor" in e.args[0]:
                raise ValueError("ActivationStatsCollector: a module ({} - {}) was encountered twice during model.apply().\n"
                                 "This is an indication that your model is using the same module instance, "
                                 "in multiple nodes in the graph.  This usually occurs with ReLU modules: \n"
                                 "For example in TorchVision's ResNet model, self.relu = nn.ReLU(inplace=True) is "
                                 "instantiated once, but used multiple times.  This is not permissible when using "
                                 "instances of ActivationStatsCollector.".
                                 format(module.distiller_name, type(module)))
            else:
                msglogger.info("Exception in _activation_stats_cb: {} {}".format(module.distiller_name, type(module)))
                raise

    def _start_counter(self, module):
        if not hasattr(module, self.stat_name):
            setattr(module, self.stat_name, AverageValueMeter())
            # Assign a name to this summary
            if hasattr(module, 'distiller_name'):
                getattr(module, self.stat_name).name = '_'.join((self.stat_name, module.distiller_name))
            else:
                getattr(module, self.stat_name).name = '_'.join((self.stat_name,
                                                                 module.__class__.__name__,
                                                                 str(id(module))))

    def _reset_counter(self, module):
        if hasattr(module, self.stat_name):
            getattr(module, self.stat_name).reset()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, self.stat_name):
            mean = getattr(module, self.stat_name).mean
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            activation_stats[getattr(module, self.stat_name).name] = mean

    def to_xlsx(self, fname):
        """Save the records to an Excel workbook, with one worksheet per layer.
        """
        fname = ".".join([fname, 'xlsx'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        with xlsxwriter.Workbook(fname) as workbook:
            worksheet = workbook.add_worksheet(self.stat_name)
            col_names = []
            for col, (module_name, module_summary_data) in enumerate(records_dict.items()):
                if not isinstance(module_summary_data, list):
                    module_summary_data = [module_summary_data]
                worksheet.write_column(1, col, module_summary_data)
                col_names.append(module_name)
            worksheet.write_row(0, 0, col_names)


class RecordsActivationStatsCollector(ActivationStatsCollector):
    """This class collects activiations statistical records.

    This Collector computes a hard-coded set of activations statsitics and collects a
    record per activation.  The activation records of the entire model (only filtered modules),
    can be saved to an Excel workbook.

    For obvious reasons, this is slower than SummaryActivationStatsCollector.
    """
    def __init__(self, model, classes=[torch.nn.ReLU]):
        super(RecordsActivationStatsCollector, self).__init__(model, "statsitics_records", classes)

    def _activation_stats_cb(self, module, input, output):
        """Record the activation sparsity of 'module'

        This is a callback from the forward() of 'module'.
        """
        def to_np(stats):
            if isinstance(stats, tuple):
                return stats[0].detach().cpu().numpy()
            else:
                return stats.detach().cpu().numpy()

        # We get a batch of activations, from which we collect statistics
        act = output.view(output.size(0), -1)
        batch_min_list = to_np(torch.min(act, dim=1)).tolist()
        batch_max_list = to_np(torch.max(act, dim=1)).tolist()
        batch_mean_list = to_np(torch.mean(act, dim=1)).tolist()
        batch_std_list = to_np(torch.std(act, dim=1)).tolist()
        batch_l2_list = to_np(torch.norm(act, p=2, dim=1)).tolist()

        module.statsitics_records['min'].extend(batch_min_list)
        module.statsitics_records['max'].extend(batch_max_list)
        module.statsitics_records['mean'].extend(batch_mean_list)
        module.statsitics_records['std'].extend(batch_std_list)
        module.statsitics_records['l2'].extend(batch_l2_list)
        module.statsitics_records['shape'] = distiller.size2str(output)

    @staticmethod
    def _create_records_dict():
        records = OrderedDict()
        for stat_name in ['min', 'max', 'mean', 'std', 'l2']:
            records[stat_name] = []
        records['shape'] = ''
        return records

    def to_xlsx(self, fname):
        """Save the records to an Excel workbook, with one worksheet per layer.
        """
        fname = ".".join([fname, 'xlsx'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        with xlsxwriter.Workbook(fname) as workbook:
            for module_name, module_act_records in records_dict.items():
                worksheet = workbook.add_worksheet(module_name)
                col_names = []
                for col, (col_name, col_data) in enumerate(module_act_records.items()):
                    if col_name == 'shape':
                        continue
                    worksheet.write_column(1, col, col_data)
                    col_names.append(col_name)
                worksheet.write_row(0, 0, col_names)
                worksheet.write(0, len(col_names)+2, module_act_records['shape'])

    def _start_counter(self, module):
        if not hasattr(module, "statsitics_records"):
            module.statsitics_records = self._create_records_dict()

    def _reset_counter(self, module):
        if hasattr(module, "statsitics_records"):
            module.statsitics_records = self._create_records_dict()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, "statsitics_records"):
            activation_stats[module.distiller_name] = module.statsitics_records


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
        yield collectors_dict
        return
    for collector in collectors_dict.values():
        collector.reset().start()
    yield collectors_dict
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
