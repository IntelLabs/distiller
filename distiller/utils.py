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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
from copy import deepcopy
import yaml
from collections import OrderedDict
import argparse
import operator


def model_device(model):
    """Determine the device the model is allocated on."""
    # Source: https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    if next(model.parameters()).is_cuda:
        return 'cuda'
    return 'cpu'


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


class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def assign_layer_fq_names(container, name=None):
    """Assign human-readable names to the modules (layers).

    Sometimes we need to access modules by their names, and we'd like to use
    fully-qualified names for convinience.
    """
    for name, module in container.named_modules():
        module.distiller_name = name


def find_module_by_fq_name(model, fq_mod_name):
    """Given a module's fully-qualified name, find the module in the provided model.

    A fully-qualified name is assigned to modules in function assign_layer_fq_names.

    Arguments:
        model: the model to search
        fq_mod_name: the module whose name we want to look up

    Returns:
        The module or None, if the module was not found.
    """
    for module in model.modules():
        if hasattr(module, 'distiller_name') and fq_mod_name == module.distiller_name:
            return module
    return None


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
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError


def density(tensor):
    """Computes the density of a tensor.

    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.

    Args:
        tensor: the tensor for which we compute the density.

    Returns:
        density (float)
    """
    # Using torch.nonzero(tensor) can lead to memory exhaustion on
    # very large tensors, so we count zeros "manually".
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


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


def sparsity_blocks(tensor, block_shape):
    """Block-wise sparsity for 4D tensors

    Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1
    """
    if tensor.dim() != 4:
        raise ValueError("sparsity_blocks is only supported for 4-D tensors")

    if len(block_shape) != 4:
        raise ValueError("Block shape must be specified as a 4-element tuple")
    block_repetitions, block_depth, block_height, block_width = block_shape
    if not block_width == block_height == 1:
        raise ValueError("Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1")

    super_block_volume = volume(block_shape)
    num_super_blocks = volume(tensor) / super_block_volume

    num_filters, num_channels = tensor.size(0), tensor.size(1)
    kernel_size = tensor.size(2) * tensor.size(3)

    # Create a view where each block is a column
    if block_depth > 1:
        view_dims = (
            num_filters*num_channels//(block_repetitions*block_depth),
            block_repetitions*block_depth,
            kernel_size,
            )
    else:
        view_dims = (
            num_filters // block_repetitions,
            block_repetitions,
            -1,
            )
    view1 = tensor.view(*view_dims)

    # Next, compute the sums of each column (block)
    block_sums = view1.abs().sum(dim=1)

    # Next, compute the sums of each column (block)
    block_sums = view1.abs().sum(dim=1)
    nonzero_blocks = len(torch.nonzero(block_sums))
    return 1 - nonzero_blocks/num_super_blocks


def sparsity_matrix(tensor, dim):
    """Generic sparsity computation for 2D matrices"""
    if tensor.dim() != 2:
        return 0

    num_structs = tensor.size()[dim]
    nonzero_structs = len(torch.nonzero(tensor.abs().sum(dim=1-dim)))
    return 1 - nonzero_structs/num_structs


def sparsity_cols(tensor, transposed=True):
    """Column-wise sparsity for 2D tensors

    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.
    In other words the matrices are stored in memory transposed.  So by default we compute
    the sparsity of the transposed dimension.
    """
    if transposed:
        return sparsity_matrix(tensor, 0)
    return sparsity_matrix(tensor, 1)


def density_cols(tensor, transposed=True):
    """Column-wise density for 2D tensors"""
    return 1 - sparsity_cols(tensor, transposed)


def sparsity_rows(tensor, transposed=True):
    """Row-wise sparsity for 2D matrices

    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.
    In other words the matrices are stored in memory transposed.  So by default we compute
    the sparsity of the transposed dimension.
    """
    if transposed:
        return sparsity_matrix(tensor, 1)
    return sparsity_matrix(tensor, 0)


def density_rows(tensor, transposed=True):
    """Row-wise density for 2D tensors"""
    return 1 - sparsity_rows(tensor, transposed)


def model_sparsity(model, param_dims=[2, 4]):
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            _density = density(param)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
    total_sparsity = (1 - sparse_params_size/params_size)*100
    return total_sparsity


def norm_filters(weights, p=1):
    """Compute the p-norm of convolution filters.

    Args:
        weights - a 4D convolution weights tensor.
                  Has shape = (#filters, #channels, k_w, k_h)
        p - the exponent value in the norm formulation
    """
    assert weights.dim() == 4
    return weights.view(weights.size(0), -1).norm(p=p, dim=1)


def model_numel(model, param_dims=[2, 4]):
    """Count the number elements in a model's parameter tensors"""
    total_numel = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            total_numel += torch.numel(param)
    return total_numel


def activation_channels_l1(activation):
    """Calculate the L1-norms of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    When the activations are computed on a distributed GPU system, different parts of the
    activation tensor might be computed by a differnt GPU. If this function is called from
    the forward-callback of some activation module in the graph, we will only witness part
    of the batch.  For example, if the batch_size is 256, and we are using 4 GPUS, instead
    of seeing activations with shape = (256, num_channels, h, w), we may see 4 calls with
    shape = (64, num_channels, h, w).

    Since we want to calculate the average of the L1-norm of each of the channels of the
    activation, we need to move the partial sums results to the CPU, where they will be
    added together.

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L! magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_norms = view_2d.norm(p=1, dim=1)  # (batch*channels) x 1
        featuremap_norms_mat = featuremap_norms.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_norms_mat = activation.norm(p=1, dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_l1: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_norms_mat.mean(dim=0).cpu()


def activation_channels_means(activation):
    """Calculate the mean of each of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L1 magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_means = view_2d.mean(dim=1)  # (batch*channels) x 1
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_means_mat.mean(dim=0).cpu()


def activation_channels_apoz(activation):
    """Calculate the APoZ of each of an activation's channels.

    APoZ is the Average Percentage of Zeros (or simply: average sparsity) and is defined in:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures".

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its sparsity.
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()


def log_training_and_weights_dist(stats_dict, params_dict,
        epoch, steps_completed, step_in_current_epoch, *loggers,
        train_steps_per_epoch=None, log_period=1):
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
        steps_completed: training steps since the begining of time
        step_in_current_epoch: training steps in current epoch
        loggers [logger.DataLogger iterable]: logger(s) to send the log info to
        train_steps_per_epoch: number of mini-batches in current epoch (used by Pylogger exclusively)
        log_period: The number of steps between logging records
    """
    for logger in loggers:
        logger.log_training_progress(
            stats_dict, epoch, steps_completed, step_in_current_epoch,
            train_steps_per_epoch=train_steps_per_epoch)
        logger.log_weights_distribution(params_dict, steps_completed)


def log_activation_statsitics(epoch, phase, loggers, collector):
    """Log information about the sparsity of the activations"""
    if collector is None:
        return
    for logger in loggers:
        logger.log_activation_statsitic(phase, collector.stat_name, collector.value(), epoch)


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


def get_dummy_input(dataset):
    if dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224)
    elif dataset == 'cifar10':
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("dataset %s is not supported" % dataset)
    return dummy_input


def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.

    torch.nn.DataParallel instances are removed.
    """
    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model


def set_deterministic():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Function to load YAML file using an OrderedDict
    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def float_range_argparse_checker(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError(
            'Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker
