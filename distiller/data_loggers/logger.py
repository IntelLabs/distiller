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

"""Loggers frontends and backends.

- DataLogger is the generic logger interface.
- PythonLogger logs using the Python logger.
- TensorBoardLogger logs to files that can be read by Google's TensorBoard.
- CsvLogger logs to CSV files.

Note that not all loggers implement all logging methods.
"""

import torch
import numpy as np
import tabulate
import distiller
from distiller.utils import density, sparsity, sparsity_2D, size_to_str, to_np, norm_filters
# TensorBoard logger
from .tbbackend import TBBackend
# Visdom logger
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import csv
import logging
msglogger = logging.getLogger()

__all__ = ['PythonLogger', 'TensorBoardLogger', 'CsvLogger']


class DataLogger(object):
    """This is an abstract interface for data loggers

    Data loggers log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """
    def __init__(self):
        pass

    def log_training_progress(self, model, epoch, i, set_size, batch_time, data_time, classerr, losses, print_freq, collectors):
        raise NotImplementedError

    def log_activation_statsitic(self, phase, stat_name, activation_stats, epoch):
        raise NotImplementedError

    def log_weights_sparsity(self, model, epoch):
        raise NotImplementedError

    def log_weights_distribution(self, named_params, steps_completed):
        pass


class PythonLogger(DataLogger):
    def __init__(self, logger):
        super(PythonLogger, self).__init__()
        self.pylogger = logger

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        stats_dict = stats_dict[1]
        if epoch > -1:
            log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed, int(total))
        else:
            log = 'Test: [{:5d}/{:5d}]    '.format(completed, int(total))
        for name, val in stats_dict.items():
            if isinstance(val, int):
                log = log + '{name} {val}    '.format(name=name, val=distiller.pretty_int(val))
            else:
                log = log + '{name} {val:.6f}    '.format(name=name, val=val)
        self.pylogger.info(log)

    def log_activation_statsitic(self, phase, stat_name, activation_stats, epoch):
        data = []
        for layer, statistic in activation_stats.items():
            data.append([layer, statistic])
        t = tabulate.tabulate(data, headers=['Layer', stat_name], tablefmt='psql', floatfmt=".2f")
        msglogger.info('\n' + t)

    def log_weights_sparsity(self, model, epoch):
        t, total = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        msglogger.info("\nParameters:\n" + str(t))
        msglogger.info('Total sparsity: {:0.2f}\n'.format(total))


class TensorBoardLogger(DataLogger):
    def __init__(self, logdir):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.tblogger = TBBackend(logdir)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False      # True
        self.logged_params = ['weight'] # ['weight', 'bias']

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        def total_steps(total, epoch, completed):
            return total*epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.tblogger.scalar_summary(prefix+tag, value, total_steps(total, epoch, completed))
        self.tblogger.sync_to_file()

    def log_activation_statsitic(self, phase, stat_name, activation_stats, epoch):
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.tblogger.scalar_summary(group+tag, value, epoch)
        self.tblogger.sync_to_file()

    def log_weights_sparsity(self, model, epoch):
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 4]:
                _density = density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.tblogger.scalar_summary('sparsity/weights/' + name,
                                             sparsity(param)*100, epoch)
                self.tblogger.scalar_summary('sparsity-2D/weights/' + name,
                                             sparsity_2D(param)*100, epoch)

        self.tblogger.scalar_summary("sparsity/weights/total", 100*(1 - sparse_params_size/params_size), epoch)
        self.tblogger.sync_to_file()

    def log_weights_filter_magnitude(self, model, epoch, multi_graphs=False):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.tblogger.list_summary('magnitude/filters/' + name,
                                           list(to_np(norm_filters(param))), epoch, multi_graphs)
        self.tblogger.sync_to_file()

    def log_weights_distribution(self, named_params, steps_completed):
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.tblogger.histogram_summary(tag, to_np(value), steps_completed)
            if self.log_gradients:
                self.tblogger.histogram_summary(tag+'/grad', to_np(value.grad), steps_completed)
        self.tblogger.sync_to_file()


class CsvLogger(DataLogger):
    def __init__(self, fname):
        super(CsvLogger, self).__init__()
        self.fname = fname

    def log_weights_sparsity(self, model, epoch):
        with open(self.fname, 'w') as csv_file:
            params_size = 0
            sparse_params_size = 0

            writer = csv.writer(csv_file)
            # write the header
            writer.writerow(['parameter', 'shape', 'volume', 'sparse volume', 'sparsity level'])

            for name, param in model.state_dict().items():
                if param.dim() in [2, 4]:
                    _density = density(param)
                    params_size += torch.numel(param)
                    sparse_params_size += param.numel() * _density
                    writer.writerow([name, size_to_str(param.size()),
                                     torch.numel(param),
                                     int(_density * param.numel()),
                                     (1-_density)*100])
