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

"""Model thinning support.

Thinning a model is the process of taking a dense network architecture with a parameter model that
has structure-sparsity (filters or channels) in the weights tensors of convolution layers, and making changes
 in the network architecture and parameters, in order to completely remove the structures.
The new architecture is smaller (condensed), with less channels and filters in some of the convolution layers.
Linear and BatchNormalization layers are also adjusted as required.

To perform thinning, we create a SummaryGraph (‘sgraph’) of our model.  We use the ‘sgraph’ to infer the
data-dependency between the modules in the PyTorch network.  This entire process is not trivial and will be
documented in a different place.
"""

import math
import logging
import copy
import re
from collections import namedtuple
import torch
from .policy import ScheduledTrainingPolicy
import distiller
from apputils import SummaryGraph
from models import ALL_MODEL_NAMES, create_model
msglogger = logging.getLogger()

ThinningRecipe = namedtuple('ThinningRecipe', ['modules', 'parameters'])
"""A ThinningRecipe is composed of two sets of instructions.
1. Instructions for setting module attributes (e.g. Conv2d.out_channels).  This set
is called 'ThinningRecipe.modules'.

2. Information on how to select specific dimensions from parameter tensors.  This
set is called 'ThinningRecipe.parameters'.


ThinningRecipe.modules is a dictionary keyed by the module names (strings).  Values
are called 'module-directives', and are grouped in another dictionary, whose keys are
the module attributes.  For example:
    features.module.19:
        in_channels: 231
        out_channels: 512
    classifier.0:
        in_channels: 22589

ThinningRecipe.parameters is a dictionary keyed by the parameter names (strings).
Values are called 'parameter directives', and each directive is a list of tuples.
These tuples can have 2 values, or 4 values.
2-value tuples have the format: (dimension-to-change, indices-to-select)
"""

__all__ = ['ThinningRecipe', 'resnet_cifar_remove_layers',
           'ChannelRemover', 'remove_channels',
           'FilterRemover',  'remove_filters',
           'execute_thinning_recipes_list']

def create_graph(dataset, arch):
    if dataset == 'imagenet':
        dummy_input = torch.randn((1, 3, 224, 224), requires_grad=False)
    elif dataset == 'cifar10':
        dummy_input = torch.randn((1, 3, 32, 32))
    assert dummy_input is not None, "Unsupported dataset ({}) - aborting draw operation".format(dataset)

    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None
    return SummaryGraph(model, dummy_input)


def normalize_layer_name(layer_name):
    start = layer_name.find('module.')
    normalized_layer_name = layer_name
    if start != -1:
        normalized_layer_name = layer_name[:start] + layer_name[start + len('module.'):]
    return normalized_layer_name


def denormalize_layer_name(model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    ugly = [mod_name for mod_name, _ in model.named_modules() if normalize_layer_name(mod_name) == normalized_name]
    assert len(ugly) == 1
    return ugly[0]


def param_name_2_layer_name(param_name):
    return param_name[:-len('weights')]


def append_param_directive(thinning_recipe, param_name, directive):
    param_directive = thinning_recipe.parameters.get(param_name, [])
    param_directive.append(directive)
    thinning_recipe.parameters[param_name] = param_directive


def append_module_directive(thinning_recipe, module_name, key, val):
    mod_directive = thinning_recipe.modules.get(module_name, {})
    mod_directive[key] = val
    thinning_recipe.modules[module_name] = mod_directive


# def merge_module_directives(dict1, dict2):
#     merged = dict1
#     for k,v2 in dict2.items():
#         v1 = dict1.get(k, {})
#         #dict2[k] = v1 + v2
#         dict2[k] = {**v1, **v2}
#     return dict2
# def merge_module_directives(dict1, dict2):
#     merged = dict1
#     for k,v2 in dict2.items():
#         v1 = dict1.get(k, {})
#         # Merge the dictionaries of attributes
#         merged[k] = {**v1, **v2}
#     return merged
#
# def merge_parameter_directives(dict1, dict2):
#     merged = dict1
#     for k,v2 in dict2.items():
#         v1 = dict1.get(k, [])
#         # Merge the lists of parameter directives
#         merged[k] = v1 + v2
#     return merged

def bn_thinning(thinning_recipe, layers, bn_name, len_thin_features, thin_features):
    """Adjust the sizes of the parameters of a BatchNormalization layer

    This function is invoked after the Convolution layer preceeding a BN layer has
    changed dimensions (filters or channels were removed), and the BN layer also
    requires updating as a result.
    """
    bn_module = layers[bn_name]
    assert isinstance(bn_module, torch.nn.modules.batchnorm.BatchNorm2d)

    bn_directive = thinning_recipe.modules.get(bn_name, {})
    bn_directive['num_features'] = len_thin_features
    # These are tensors that BN uses for temporary storage of batch statistics.
    # The dimensions of these tensors need adjustment, by removing specific elements
    # from the tensors.
    bn_directive['running_mean'] = (0, thin_features)
    bn_directive['running_var'] = (0, thin_features)
    thinning_recipe.modules[bn_name] = bn_directive

    # These are the scale and shift tensors
    thinning_recipe.parameters[bn_name+'.weight'] = [(0, thin_features)]
    thinning_recipe.parameters[bn_name+'.bias'] = [(0, thin_features)]


def resnet_cifar_remove_layers(model):
    """Remove layers from ResNet-Cifar

    Search for convolution layers which have 100% sparse weight tensors and remove
    them from the model.  This ugly code is specific to ResNet for Cifar, using the
    layer gating mechanism that we added in order to remove layers from the network.
    """

    # Create a list of all the layers that have their weights tensor 100% sparse
    layers_to_remove = [param_name for param_name, param in model.named_parameters()
                        if distiller.density(param) == 0]

    for param_name in layers_to_remove:
        parts = param_name.split('.')
        layer = int(parts[1][-1]) - 1
        block = int(parts[2])
        if parts[3] == 'downsample':
            downsample = int(parts[4][0])
            msglogger.info("Removing layer: %s [layer=%d block=%d downsample=%d]" %
                           (param_name[:param_name.find('.weight')], layer, block, downsample))
        else:
            conv = int(parts[3][-1]) - 1
            msglogger.info("Removing layer: %s [layer=%d block=%d conv=%d]" %
                           (param_name[:param_name.find('.weight')], layer, block, conv))

        model.module.layer_gates[layer][block][conv] = False


def remove_channels(model, zeros_mask_dict, arch, dataset):
    sgraph = create_graph(dataset, arch)
    thinning_recipe = create_thinning_recipe_channels(sgraph, model, zeros_mask_dict)
    apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe)
    return model


def find_nonzero_channels(param, param_name):
    """Count the number of non-zero channels in a weights tensor.

    Non-zero channels are channels that have at least one coefficient that is
    non-zero.  Counting non-zero channels involves some tensor acrobatics.
    """

    num_filters = param.size(0)
    num_channels = param.size(1)

    # First, reshape the weights tensor such that each channel (kernel) in the original
    # tensor, is now a row in the 2D tensor.
    view_2d = param.view(-1, param.size(2) * param.size(3))
    # Next, compute the sums of each kernel
    kernel_sums = view_2d.abs().sum(dim=1)
    # Now group by channels
    k_sums_mat = kernel_sums.view(num_filters, num_channels).t()
    nonzero_channels = torch.nonzero(k_sums_mat.abs().sum(dim=1))

    if num_channels > len(nonzero_channels):
        msglogger.info("In tensor %s found %d/%d zero channels", param_name,
                       num_filters - len(nonzero_channels), num_filters)

    return nonzero_channels


def apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe):
    if len(thinning_recipe.modules)>0 or len(thinning_recipe.parameters)>0:
        # Now actually remove the filters, chaneels and make the weight tensors smaller
        execute_thinning_recipe(model, zeros_mask_dict, thinning_recipe)

        # Stash the recipe, so that it will be serialized together with the model
        if hasattr(model, 'thinning_recipes'):
            # Merge the new recipe with the exiting recipe.  Modules and Parameter
            # dictionaries are merge separately.
            # merged_recipes = ThinningRecipe(
            #                   modules = merge_module_directives(model.thinning_recipe.modules,
            #                                                     thinning_recipe.modules),
            #                   parameters = merge_parameter_directives(model.thinning_recipe.parameters,
            #                                                           thinning_recipe.parameters))
            model.thinning_recipes.append(thinning_recipe)
        else:
            model.thinning_recipes = [thinning_recipe]
        msglogger.info("Created, applied and saved a filter-thinning recipe")
    else:
        msglogger.error("Failed to create a thinning recipe")


def remove_filters(model, zeros_mask_dict, arch, dataset):
    sgraph = create_graph(dataset, arch)
    thinning_recipe = create_thinning_recipe_filters(sgraph, model, zeros_mask_dict)
    apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe)
    return model


def create_thinning_recipe_channels(sgraph, model, zeros_mask_dict):
    """Create a recipe for removing channels from Convolution layers.

    The 4D weights of the model parameters (i.e. the convolution parameters) are
    examined one by one, to determine which has channels that are all zeros.
    For each weights tensor that has at least one zero-channel, we create a
    "thinning recipe".
    The thinning recipe contains meta-instructions of how the model
    should be changed in order to remove the channels.
    """
    msglogger.info("Invoking create_thinning_recipe_channels")

    thinning_recipe = ThinningRecipe(modules={}, parameters={})
    layers = {mod_name : m for mod_name, m in model.named_modules()}

    # Traverse all of the model's parameters, search for zero-channels, and
    # create a thinning recipe that descibes the required changes to the model.
    for param_name, param in model.named_parameters():
        # We are only interested in 4D weights (of Convolution layers)
        if param.dim() != 4:
            continue

        num_channels = param.size(1)
        nonzero_channels = find_nonzero_channels(param, param_name)

        # If there are non-zero channels in this tensor then continue to next tensor
        if num_channels <= len(nonzero_channels):
            continue

        # We are removing channels, so update the number of incoming channels (IFMs)
        # in the convolutional layer
        layer_name = param_name_2_layer_name(param_name)
        assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
        append_module_directive(thinning_recipe, layer_name, key='in_channels', val=len(nonzero_channels))

        # Select only the non-zero filters
        indices = nonzero_channels.data.squeeze()
        append_param_directive(thinning_recipe, param_name, (1, indices))

        # Find all instances of Convolution layers that immediately preceed this layer
        predecessors = sgraph.predecessors_f(normalize_layer_name(layer_name), ['Conv'])
        # Convert the layers names to PyTorch's convoluted naming scheme (when DataParallel is used)
        predecessors = [denormalize_layer_name(model, predecessor) for predecessor in predecessors]
        for predecessor in predecessors:
            # For each of the convolutional layers that preceed, we have to reduce the number of output channels.
            append_module_directive(thinning_recipe, layer_name, key='out_channels', val=len(nonzero_channels))

            # Now remove channels from the weights tensor of the successor conv
            append_param_directive(thinning_recipe, predecessor+'.weight', (0, indices))

        # Now handle the BatchNormalization layer that follows the convolution
        bn_layers = sgraph.predecessors_f(normalize_layer_name(layer_name), ['BatchNormalization'])
        if len(bn_layers) > 0:
            assert len(bn_layers) == 1
            # Thinning of the BN layer that follows the convolution
            bn_layer_name = denormalize_layer_name(model, bn_layers[0])
            bn_thinning(thinning_recipe, layers, bn_layer_name, len_thin_features=len(nonzero_channels), thin_features=indices)

    return thinning_recipe


def create_thinning_recipe_filters(sgraph, model, zeros_mask_dict):
    """Create a recipe for removing filters from Convolution layers.

    The 4D weights of the model parameters (i.e. the convolution parameters) are
    examined one by one, to determine which has filters that are all zeros.
    For each weights tensor that has at least one zero-filter, we create a
    "thinning recipe".
    The thinning recipe contains meta-instructions of how the model
    should be changed in order to remove the filters.
    """
    msglogger.info("Invoking create_thinning_recipe_filters")

    thinning_recipe = ThinningRecipe(modules={}, parameters={})
    layers = {mod_name : m for mod_name, m in model.named_modules()}

    for param_name, param in model.named_parameters():
        # We are only interested in 4D weights
        if param.dim() != 4:
            continue

        # Find the number of zero-valued filters in this weights tensor
        filter_view = param.view(param.size(0), -1)
        num_filters = filter_view.size()[0]
        nonzero_filters = torch.nonzero(filter_view.abs().sum(dim=1))

        # If there are non-zero filters in this tensor then continue to next tensor
        if num_filters <= len(nonzero_filters):
            msglogger.debug("SKipping {} shape={}".format(param_name_2_layer_name(param_name), param.shape))
            continue

        msglogger.info("In tensor %s found %d/%d zero filters", param_name,
                       num_filters - len(nonzero_filters), num_filters)

        # We are removing filters, so update the number of outgoing channels (OFMs)
        # in the convolutional layer
        layer_name = param_name_2_layer_name(param_name)
        assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
        append_module_directive(thinning_recipe, layer_name, key='out_channels', val=len(nonzero_filters))

        # Select only the non-zero filters
        indices = nonzero_filters.data.squeeze()
        append_param_directive(thinning_recipe, param_name, (0, indices))

        if layers[layer_name].bias is not None:
            # This convolution has bias coefficients
            append_param_directive(thinning_recipe, layer_name+'.bias', (0, indices))

        # Find all instances of Convolution or FC (GEMM) layers that immediately follow this layer
        successors = sgraph.successors_f(normalize_layer_name(layer_name), ['Conv', 'Gemm'])
        # Convert the layers names to PyTorch's convoluted naming scheme (when DataParallel is used)
        successors = [denormalize_layer_name(model, successor) for successor in successors]
        for successor in successors:

            if isinstance(layers[successor], torch.nn.modules.Conv2d):
                # For each of the convolutional layers that follow, we have to reduce the number of input channels.
                append_module_directive(thinning_recipe, successor, key='in_channels', val=len(nonzero_filters))
                msglogger.info("[recipe] {}: setting in_channels = {}".format(successor, len(nonzero_filters)))

                # Now remove channels from the weights tensor of the successor conv
                append_param_directive(thinning_recipe, successor+'.weight', (1, indices))

            elif isinstance(layers[successor], torch.nn.modules.Linear):
                # If a Linear (Fully-Connected) layer follows, we need to update it's in_features member
                fm_size = layers[successor].in_features // layers[layer_name].out_channels
                in_features = fm_size * len(nonzero_filters)
                #append_module_directive(thinning_recipe, layer_name, key='in_features', val=in_features)
                append_module_directive(thinning_recipe, successor, key='in_features', val=in_features)
                msglogger.info("[recipe] {}: setting in_features = {}".format(successor, in_features))

                # Now remove channels from the weights tensor of the successor FC layer:
                # This is a bit tricky:
                fm_height = fm_width = int(math.sqrt(fm_size))
                view_4D = (layers[successor].out_features, layers[layer_name].out_channels, fm_height, fm_width)
                view_2D = (layers[successor].out_features, in_features)
                append_param_directive(thinning_recipe, successor+'.weight', (1, indices, view_4D, view_2D))

        # Now handle the BatchNormalization layer that follows the convolution
        bn_layers = sgraph.successors_f(normalize_layer_name(layer_name), ['BatchNormalization'])
        if len(bn_layers) > 0:
            assert len(bn_layers) == 1
            # Thinning of the BN layer that follows the convolution
            bn_layer_name = denormalize_layer_name(model, bn_layers[0])
            bn_thinning(thinning_recipe, layers, bn_layer_name, len_thin_features=len(nonzero_filters), thin_features=indices)
    return thinning_recipe


class ChannelRemover(ScheduledTrainingPolicy):
    """A policy which applies a network thinning function"""
    def __init__(self, thinning_func_str, arch, dataset):
        self.thinning_func = globals()[thinning_func_str]
        self.arch = arch
        self.dataset = dataset

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        self.thinning_func(model, zeros_mask_dict, self.arch, self.dataset)


class FilterRemover(ScheduledTrainingPolicy):
    """A policy which applies a network thinning function"""
    def __init__(self, thinning_func_str, arch, dataset):
        self.thinning_func = globals()[thinning_func_str]
        self.arch = arch
        self.dataset = dataset
        self.done = False
        self.active_cb = "on_minibatch_begin"

    def __apply(self, model, zeros_mask_dict):
        if not self.done:
            # We want to execute the thinning function only once, not every invocation of on_minibatch_begin
            self.thinning_func(model, zeros_mask_dict, self.arch, self.dataset)
            self.done = True

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict):
        # We hook onto the on_minibatch_begin because we want to run after the pruner which sparsified
        # the tensors.  Pruners configure their pruning mask in on_epoch_begin, but apply the mask
        # only in on_minibatch_begin
        if self.active_cb != "on_minibatch_begin":
            return
        self.__apply(model, zeros_mask_dict)

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict):
        if self.active_cb != "on_minibatch_end":
            return
        self.__apply(model, zeros_mask_dict)

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        # The epoch has ended and we reset the 'done' flag, so that the FilterRemover instance can be reused
        self.done = False

def execute_thinning_recipes_list(model, zeros_mask_dict, recipe_list):
    # Invoke this function when you want to use a list of thinning recipes to convert a programmed model
    # to a thinned model. For example, this is invoked when loading a model from a checkpoint.
    for i, recipe in enumerate(recipe_list):
        msglogger.info("recipe %d" % i)
        execute_thinning_recipe(model, zeros_mask_dict, recipe, loaded_from_file=True)

def execute_thinning_recipe(model, zeros_mask_dict, recipe, loaded_from_file=False):
    """Apply a thinning recipe to a model.

    This will remove filters and channels, as well as handle batch-normalization parameter
    adjustment, and thinning of weight tensors.
    """
    layers = {}
    for name, m in model.named_modules():
        layers[name] = m

    for layer_name, directives in recipe.modules.items():
        for attr, val in directives.items():
            if attr in ['running_mean', 'running_var']:
                msglogger.info("[thinning] {}: setting {} to {}".format(layer_name, attr, len(val[1])))
                setattr(layers[layer_name], attr,
                        torch.index_select(getattr(layers[layer_name], attr),
                                           dim=val[0], index=val[1]))
            else:
                msglogger.info("[thinning] {}: setting {} to {}".format(layer_name, attr, val))
                setattr(layers[layer_name], attr, val)

    assert len(recipe.parameters) > 0

    for param_name, param_directives in recipe.parameters.items():
        param = distiller.model_find_param(model, param_name)
        for directive in param_directives:
            dim = directive[0]
            indices = directive[1]
            if len(directive) == 4:  # TODO: this code is hard to follow
                selection_view = param.view(*directive[2])
                param.data = torch.index_select(selection_view, dim, indices)

                if param.grad is not None:
                    # We also need to change the dimensions of the gradient tensor.
                    grad_selection_view = param.grad.resize_(*directive[2])
                    param.grad = torch.index_select(grad_selection_view, dim, indices)

                param.data = param.view(*directive[3])
                if param.grad is not None:
                    param.grad = param.grad.resize_(*directive[3])
            else:
                param.data = torch.index_select(param.data, dim, indices)
                # We also need to change the dimensions of the gradient tensor.
                # If have not done a backward-pass thus far, then the gradient will
                # not exist, and therefore won't need to be re-dimensioned.
                if param.grad is not None:
                    param.grad = torch.index_select(param.grad, dim, indices)
                msglogger.info("[thinning] changing param {} shape: {}".format(param_name, len(indices)))

            if not loaded_from_file:
                # If the masks are loaded from a checkpoint file, then we don't need to change
                # their shape, because they are already correctly shaped
                mask = zeros_mask_dict[param_name].mask
                if mask is not None: # and (mask.size(dim) != len(indices)):
                    zeros_mask_dict[param_name].mask = torch.index_select(mask, dim, indices)
