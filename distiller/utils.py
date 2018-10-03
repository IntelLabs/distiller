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

"""A collection of useful utility functions.

This module contains various tensor sparsity/density measurement functions, together
with some random helper functions.
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from copy import deepcopy


def to_np(var):
    return var.data.cpu().numpy()


def size2str(torch_size):
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, torch.FloatTensor) or isinstance(torch_size, torch.cuda.FloatTensor):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    raise TypeError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'


def pretty_int(i):
    return "{:,}".format(i)


def normalize_module_name(layer_name):
    """Normalize a module's name.

    PyTorch let's you parallelize the computation of a model, by wrapping a model with a
    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,
    even though the actual functionality of the module doesn't change.
    Many time, when we search for modules by name, we are indifferent to the DataParallel
    module and want to use the same module name whether the module is parallel or not.
    We call this module name normalization, and this is implemented here.
    """
    if layer_name.find("module.") >= 0:
        return layer_name.replace("module.", "")
    return layer_name.replace(".module", "")


def denormalize_module_name(parallel_model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name   # Did not find a module with the name <normalized_name>


def volume(tensor):
    """return the volume of a pytorch tensor"""
    return np.prod(tensor.shape)


def density(tensor):
    """Computes the density of a tensor.

    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.

    Args:
        tensor: the tensor for which we compute the density.

    Returns:
        density (float)
    """
    assert torch.numel(tensor) > 0
    nonzero = torch.nonzero(tensor)
    if nonzero.dim() == 0:
        return 0.0
    return nonzero.size(0) / float(torch.numel(tensor))


def sparsity(tensor):
    """Computes the sparsity of a tensor.

    Sparsity is the fraction of zero elements in a tensor.
    If a tensor has a density of 0.0, then it has all zero elements.
    Sparsity and density are complementary.

    Args:
        tensor: the tensor for which we compute the density.

    Returns:
        sparsity (float)
    """
    return 1.0 - density(tensor)


def sparsity_3D(tensor):
    """Filter-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0
    view_3d = tensor.view(-1, tensor.size(1) * tensor.size(2) * tensor.size(3))
    num_filters = view_3d.size()[0]
    nonzero_filters = len(torch.nonzero(view_3d.abs().sum(dim=1)))
    return 1 - nonzero_filters/num_filters


def density_3D(tensor):
    """Filter-wise density for 4D tensors"""
    return 1 - sparsity_3D(tensor)


def sparsity_2D(tensor):
    """Create a list of sparsity levels for each channel in the tensor 't'

    For 4D weight tensors (convolution weights), we flatten each kernel (channel)
    so it becomes a row in a 3D tensor in which each channel is a filter.
    So if the original 4D weights tensor is:
        #OFMs x #IFMs x K x K
    The flattened tensor is:
        #OFMS x #IFMs x K^2

    For 2D weight tensors (fully-connected weights), the tensors is shaped as
        #IFMs x #OFMs
    so we don't need to flatten anything.

    To measure 2D sparsity, we sum the absolute values of the elements in each row,
    and then count the number of rows having sum(abs(row values)) == 0.
    """
    if tensor.dim() == 4:
        # For 4D weights, 2D structures are channels (filter kernels)
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        # For 2D weights, 2D structures are either columns or rows.
        # At the moment, we only support row structures
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs/num_structs


def density_2D(tensor):
    """Kernel-wise sparsity for 4D tensors"""
    return 1 - sparsity_2D(tensor)


def sparsity_ch(tensor):
    """Channel-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0

    num_filters = tensor.size(0)
    num_kernels_per_filter = tensor.size(1)

    # First, reshape the weights tensor such that each channel (kernel) in the original
    # tensor, is now a row in the 2D tensor.
    view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    # Next, compute the sums of each kernel
    kernel_sums = view_2d.abs().sum(dim=1)
    # Now group by channels
    k_sums_mat = kernel_sums.view(num_filters, num_kernels_per_filter).t()
    nonzero_channels = len(torch.nonzero(k_sums_mat.abs().sum(dim=1)))
    return 1 - nonzero_channels/num_kernels_per_filter


def density_ch(tensor):
    """Channel-wise density for 4D tensors"""
    return 1 - sparsity_ch(tensor)


def sparsity_cols(tensor):
    """Column-wise sparsity for 2D tensors"""
    if tensor.dim() != 2:
        return 0

    num_cols = tensor.size()[1]
    nonzero_cols = len(torch.nonzero(tensor.abs().sum(dim=0)))
    return 1 - nonzero_cols/num_cols


def density_cols(tensor):
    """Column-wise density for 2D tensors"""
    return 1 - sparsity_cols(tensor)


def sparsity_rows(tensor):
    """Row-wise sparsity for 2D matrices"""
    if tensor.dim() != 2:
        return 0

    num_rows = tensor.size()[0]
    nonzero_rows = len(torch.nonzero(tensor.abs().sum(dim=1)))
    return 1 - nonzero_rows/num_rows


def density_rows(tensor):
    """Row-wise density for 2D tensors"""
    return 1 - sparsity_rows(tensor)


def model_numel(model, param_dims=[2, 4]):
    """Count the number elements in a model's parameter tensors"""
    total_numel = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            total_numel += torch.numel(param)
    return total_numel


def log_training_progress(stats_dict, params_dict, epoch, steps_completed, total_steps, log_freq, loggers):
    """Log information about the training progress, and the distribution of the weight tensors.

    Args:
        stats_dict: A tuple of (group_name, dict(var_to_log)).  Grouping statistics variables is useful for logger
          backends such as TensorBoard.  The dictionary of var_to_log has string key, and float values.
          For example:
              stats = ('Peformance/Validation/',
                       OrderedDict([('Loss', vloss),
                                    ('Top1', top1),
                                    ('Top5', top5)]))
        params_dict: A parameter dictionary, such as the one returned by model.named_parameters()
        epoch: The current epoch
        steps_completed: The current step in the epoch
        total_steps: The total number of training steps taken so far
        log_freq: The number of steps between logging records
        loggers: A list of loggers to send the log info to
    """
    for logger in loggers:
        logger.log_training_progress(stats_dict, epoch,
                                     steps_completed,
                                     total_steps, freq=log_freq)
        logger.log_weights_distribution(params_dict, steps_completed)


def log_activation_sparsity(epoch, loggers, collector):
    """Log information about the sparsity of the activations"""
    for logger in loggers:
        logger.log_activation_sparsity(collector.value(), epoch)


def log_weights_sparsity(model, epoch, loggers):
    """Log information about the weights sparsity"""
    for logger in loggers:
        logger.log_weights_sparsity(model, epoch)


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


class DoNothingModuleWrapper(nn.Module):
    """Implement a nn.Module which wraps another nn.Module.

    The DoNothingModuleWrapper wrapper does nothing but forward
    to the wrapped module.
    One use-case for this class, is for replacing nn.DataParallel
    by a module that does nothing :-).  This is a trick we use
    to transform data-parallel to serialized models.
    """
    def __init__(self, module):
        super(DoNothingModuleWrapper, self).__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.

    nn.DataParallel instances are replaced by DoNothingModuleWrapper
    instances.
    """
    def replace_data_parallel(container, prefix=''):
        for name, module in container.named_children():
            full_name = prefix + name
            if isinstance(module, nn.DataParallel):
                # msglogger.debug('Replacing module {}'.format(full_name))
                setattr(container, name, DoNothingModuleWrapper(module.module))
            if has_children(module):
                # For a container we call recursively
                replace_data_parallel(module, full_name + '.')

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        # new_model = new_model.module #
        new_model = DoNothingModuleWrapper(new_model.module)

    replace_data_parallel(new_model)
    return new_model
