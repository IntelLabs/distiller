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

"""Model statistics summaries.

    - weights sparsities
    - optimizer state
    - model details
"""
from functools import partial
import pandas as pd
from tabulate import tabulate
import logging
import torch
from torch.autograd import Variable
import torch.optim
import distiller
from .data_loggers import PythonLogger, CsvLogger

msglogger = logging.getLogger()

__all__ = ['model_summary',
           'weights_sparsity_summary', 'weights_sparsity_tbl_summary',
           'model_performance_summary', 'model_performance_tbl_summary']


def model_summary(model, what, dataset=None):
    if what == 'sparsity':
        pylogger = PythonLogger(msglogger)
        csvlogger = CsvLogger('weights.csv')
        distiller.log_weights_sparsity(model, -1, loggers=[pylogger, csvlogger])
    elif what == 'compute':
        if dataset == 'imagenet':
            dummy_input = Variable(torch.randn(1, 3, 224, 224))
        elif dataset == 'cifar10':
            dummy_input = Variable(torch.randn(1, 3, 32, 32))
        else:
            print("Unsupported dataset (%s) - aborting compute operation" % dataset)
            return
        df = model_performance_summary(model, dummy_input, 1)
        t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
        total_macs = df['MACs'].sum()
        print(t)
        print("Total MACs: " + "{:,}".format(total_macs))
    elif what == 'model':
        # print the simple form of the model
        print(model)
    elif what == 'modules':
        # Print the names of non-leaf modules
        # Remember that in PyTorch not every node is a module (e.g. F.relu).
        # Also remember that parameterless modules, like nn.MaxPool2d, can be used multiple
        # times in the same model, but they will only appear once in the modules list.
        nodes = []
        for name, module in model.named_modules():
            # Only print leaf modules
            if len(module._modules) == 0:
                nodes.append([name, module.__class__.__name__])
        print(tabulate(nodes, headers=['Name', 'Type']))
    else:
        raise ValueError("%s is not a supported summary type" % what)


def weights_sparsity_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)',
                               'Cols (%)', 'Rows (%)', 'Ch (%)', '2D (%)', '3D (%)',
                               'Fine (%)', 'Std', 'Mean', 'Abs-Mean'])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            _density = distiller.density(param)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([
                name,
                distiller.size_to_str(param.size()),
                torch.numel(param),
                int(_density * param.numel()),
                distiller.sparsity_cols(param)*100,
                distiller.sparsity_rows(param)*100,
                distiller.sparsity_ch(param)*100,
                distiller.sparsity_2D(param)*100,
                distiller.sparsity_3D(param)*100,
                (1-_density)*100,
                param.std().item(),
                param.mean().item(),
                param.abs().mean().item()
            ])

    total_sparsity = (1 - sparse_params_size/params_size)*100

    df.loc[len(df.index)] = ([
        'Total sparsity:',
        '-',
        params_size,
        int(sparse_params_size),
        0, 0, 0, 0, 0,
        total_sparsity,
        0, 0, 0])

    if return_total_sparsity:
        return df, total_sparsity
    return df


def weights_sparsity_tbl_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df, total_sparsity = weights_sparsity_summary(model, return_total_sparsity=True, param_dims=param_dims)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    if return_total_sparsity:
        return t, total_sparsity
    return t


# Performance data collection  code follows from here down

def conv_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Conv2d)
    if self in memo:
        return

    weights_vol = self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    macs = (distiller.volume(output) *
            (self.in_channels / self.groups * self.kernel_size[0] * self.kernel_size[1]))
    attrs = 'k=' + '('+(', ').join(['%d' % v for v in self.kernel_size])+')'
    module_visitor(self, input, output, df, model, weights_vol, macs, attrs)


def fc_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Linear)
    if self in memo:
        return

    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    weights_vol = macs = self.in_features * self.out_features
    module_visitor(self, input, output, df, model, weights_vol, macs)


def module_visitor(self, input, output, df, model, weights_vol, macs, attrs=None):
    in_features_shape = input[0].size()
    out_features_shape = output.size()

    mod_name = distiller.model_find_module_name(model, self)
    df.loc[len(df.index)] = ([mod_name, self.__class__.__name__,
                              attrs if attrs is not None else '',
                              distiller.size_to_str(in_features_shape), distiller.volume(input[0]),
                              distiller.size_to_str(out_features_shape), distiller.volume(output),
                              int(weights_vol), int(macs)])


def model_performance_summary(model, dummy_input, batch_size=1):
    """Collect performance data"""
    def install_perf_collector(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(
                                    partial(conv_visitor, df=df, model=model, memo=memo)))
        elif isinstance(m, torch.nn.Linear):
            hook_handles.append(m.register_forward_hook(
                                    partial(fc_visitor, df=df, model=model, memo=memo)))

    df = pd.DataFrame(columns=['Name', 'Type', 'Attrs', 'IFM', 'IFM volume',
                               'OFM', 'OFM volume', 'Weights volume', 'MACs'])

    hook_handles = []
    memo = []

    model = distiller.make_non_parallel_copy(model)
    model.apply(install_perf_collector)
    # Now run the forward path and collect the data
    model(dummy_input.cuda())
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return df


def model_performance_tbl_summary(model, dummy_input, batch_size):
    df = model_performance_summary(model, dummy_input, batch_size)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    return t
