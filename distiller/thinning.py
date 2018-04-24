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

import logging
import copy
import re
from collections import namedtuple
import torch
from .policy import ScheduledTrainingPolicy
import distiller
msglogger = logging.getLogger()

ThinningRecipe = namedtuple('ThinningRecipe', ['modules', 'parameters'])

__all__ = ['ThinningRecipe', 'resnet_cifar_remove_layers', 'resnet_cifar_remove_filters',
           'resnet_cifar_remove_channels', 'ResnetCifarChannelRemover',
           'ResnetCifarFilterRemover', 'execute_thinning_recipe']

# This is a dictionary that maps ResNet-Cifar connectivity, of convolutional layers.
# BN layers connectivity are implied; and shortcuts are not currently handled.
# Format:
#   layer_name : [list of follwers]
#
# This is a temporary hack.
conv_connectivity = {
    'module.conv1':          'module.layer1.0.conv1',
    'module.layer1.0.conv1': 'module.layer1.0.conv2',
    'module.layer1.0.conv2': 'module.layer1.1.conv1',
    'module.layer1.1.conv1': 'module.layer1.1.conv2',
    'module.layer1.1.conv2': 'module.layer1.2.conv1',
    'module.layer1.2.conv1': 'module.layer1.2.conv2',
    'module.layer1.2.conv2': 'module.layer1.3.conv1',
    'module.layer1.3.conv1': 'module.layer1.3.conv2',
    'module.layer1.3.conv2': 'module.layer1.4.conv1',
    'module.layer1.4.conv1': 'module.layer1.4.conv2',
    'module.layer1.4.conv2': 'module.layer1.5.conv1',
    'module.layer1.5.conv1': 'module.layer1.5.conv2',
    'module.layer1.5.conv2': 'module.layer1.6.conv1',
    'module.layer1.6.conv1': 'module.layer1.6.conv2',
    'module.layer1.6.conv2': 'module.layer1.7.conv1',
    'module.layer1.7.conv1': 'module.layer1.7.conv2',
    'module.layer1.7.conv2': 'module.layer1.8.conv1',
    'module.layer1.8.conv1': 'module.layer1.8.conv2',
    'module.layer1.8.conv2': 'module.layer2.0.conv1',

    'module.layer2.0.conv1': 'module.layer2.0.conv2',
    'module.layer2.0.conv2': 'module.layer2.1.conv1',
    'module.layer2.1.conv1': 'module.layer2.1.conv2',
    'module.layer2.1.conv2': 'module.layer2.2.conv1',
    'module.layer2.2.conv1': 'module.layer2.2.conv2',
    'module.layer2.2.conv2': 'module.layer2.3.conv1',
    'module.layer2.3.conv1': 'module.layer2.3.conv2',
    'module.layer2.3.conv2': 'module.layer2.4.conv1',
    'module.layer2.4.conv1': 'module.layer2.4.conv2',
    'module.layer2.4.conv2': 'module.layer2.5.conv1',
    'module.layer2.5.conv1': 'module.layer2.5.conv2',
    'module.layer2.5.conv2': 'module.layer2.6.conv1',
    'module.layer2.6.conv1': 'module.layer2.6.conv2',
    'module.layer2.6.conv2': 'module.layer2.7.conv1',
    'module.layer2.7.conv1': 'module.layer2.7.conv2',
    'module.layer2.7.conv2': 'module.layer2.8.conv1',
    'module.layer2.8.conv1': 'module.layer2.8.conv2',
    'module.layer2.8.conv2': 'module.layer3.0.conv1',

    'module.layer3.0.conv1': 'module.layer3.0.conv2',
    'module.layer3.0.conv2': 'module.layer3.1.conv1',
    'module.layer3.1.conv1': 'module.layer3.1.conv2',
    'module.layer3.1.conv2': 'module.layer3.2.conv1',
    'module.layer3.2.conv1': 'module.layer3.2.conv2',
    'module.layer3.2.conv2': 'module.layer3.3.conv1',
    'module.layer3.3.conv1': 'module.layer3.3.conv2',
    'module.layer3.3.conv2': 'module.layer3.4.conv1',
    'module.layer3.4.conv1': 'module.layer3.4.conv2',
    'module.layer3.4.conv2': 'module.layer3.5.conv1',
    'module.layer3.5.conv1': 'module.layer3.5.conv2',
    'module.layer3.5.conv2': 'module.layer3.6.conv1',
    'module.layer3.6.conv1': 'module.layer3.6.conv2',
    'module.layer3.6.conv2': 'module.layer3.7.conv1',
    'module.layer3.7.conv1': 'module.layer3.7.conv2',
    'module.layer3.7.conv2': 'module.layer3.8.conv1',
    'module.layer3.8.conv1': 'module.layer3.8.conv2',
    'module.layer3.8.conv2': 'module.fc'}

def find_predecessors(layer_name):
    predecessors = []
    for layer, followers in conv_connectivity.items():
        followers = follwers if isinstance(followers, list) else [followers]
        if layer_name in followers:
            predecessors.append(layer)
    return predecessors


def bn_thinning(thinning_recipe, layers, bn_name, len_thin_features, thin_features):
    """Adjust the sizes of the parameters of a BatchNormalization layer"""
    bn_module = layers[bn_name]
    assert isinstance(bn_module, torch.nn.modules.batchnorm.BatchNorm2d)

    bn_directive = thinning_recipe.modules.get(bn_name, {})
    bn_directive['num_features'] = len_thin_features

    # THIS IS A HACK
    bn_directive['running_mean'] = (0, thin_features)
    bn_directive['running_var'] = (0, thin_features)

    thinning_recipe.modules[bn_name] = bn_directive
    thinning_recipe.parameters[bn_name+'.weight'] = (0, thin_features)
    thinning_recipe.parameters[bn_name+'.bias'] = (0, thin_features)


def resnet_cifar_remove_layers(resnet_cifar_model):
    """Remove layers from ResNet-Cifar

    Search for convolution layers which have 100% sparse weight tensors and remove
    them from the model.  This ugly code is specific to ResNet for Cifar, using the
    layer gating mechanism that we added in order to remove layers from the network.
    """

    # Create a list of all the layers that have their weights tensor 100% sparse
    layers_to_remove = [param_name for param_name, param in resnet_cifar_model.named_parameters()
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

        resnet_cifar_model.module.layer_gates[layer][block][conv] = False

def resnet_cifar_remove_channels(resnet_cifar_model, zeros_mask_dict):
    thinning_recipe = resnet_cifar_create_thinning_recipe_channels(resnet_cifar_model, zeros_mask_dict)

    if len(thinning_recipe.modules) > 0 or len(thinning_recipe.parameters)>0:
        # Stash the recipe, so that it will be serialized together with the model
        resnet_cifar_model.thinning_recipe = thinning_recipe
        # Now actually remove the filters, chaneels and make the weight tensors smaller
        execute_thinning_recipe(resnet_cifar_model, zeros_mask_dict, thinning_recipe)
        msglogger.info("Created, applied and saved a channel-thinning recipe")

    return resnet_cifar_model


def resnet_cifar_create_thinning_recipe_channels(resnet_cifar_model, zeros_mask_dict):
    thinning_recipe = ThinningRecipe(modules={}, parameters={})
    layers = {thin_name : m for thin_name, m in resnet_cifar_model.named_modules()}

    for name, p_thin in resnet_cifar_model.named_parameters():
        if p_thin.dim() != 4:
            continue

        num_filters = p_thin.size(0)
        num_channels = p_thin.size(1)

        # First, reshape the weights tensor such that each channel (kernel) in the original
        # tensor, is now a row in the 2D tensor.
        view_2d = p_thin.view(-1, p_thin.size(2) * p_thin.size(3))
        # Next, compute the sums of each kernel
        kernel_sums = view_2d.abs().sum(dim=1)
        # Now group by channels
        k_sums_mat = kernel_sums.view(num_filters, num_channels).t()
        nonzero_channels = torch.nonzero(k_sums_mat.abs().sum(dim=1))

        if num_channels > len(nonzero_channels):
            msglogger.info("In tensor %s found %d/%d zero channels", name,
                           num_filters - len(nonzero_channels), num_filters)

            # Update the number of incoming channels (IFMs) in the convolutional layer
            layer_name = name[:-len('weights')]
            assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
            thinning_recipe.modules[layer_name] = {}
            thinning_recipe.modules[layer_name]['in_channels'] = len(nonzero_channels)

            # Select only the non-zero filters
            indices = nonzero_channels.data.squeeze()
            thinning_recipe.parameters[name] = (1, indices)

            assert layer_name in conv_connectivity
            predecessors = find_predecessors(layer_name)

            for predecessor in predecessors:
                # For each of the convolutional layers that preceed, we have to reduce the number of output channels.
                conv_directive = thinning_recipe.modules.get(predecessor, {})
                conv_directive['out_channels'] = len(nonzero_channels)
                thinning_recipe.modules[predecessor] = conv_directive

                # Now remove channels from the weights tensor of the follower conv
                #predecessor_param = distiller.model_find_param(resnet_cifar_model, predecessor+'.weight')
                thinning_recipe.parameters[predecessor+'.weight'] = (0, indices)

                # Now handle the BatchNormalization layer that follows the convolution
                bn_name = predecessor.replace('conv', 'bn')
                # Thinning of the BN layer that follows the convolution
                bn_thinning(thinning_recipe, layers, bn_name, len_thin_features=len(nonzero_channels), thin_features=indices)

    return thinning_recipe


def resnet_cifar_remove_filters(resnet_cifar_model, zeros_mask_dict):
    thinning_recipe = resnet_cifar_create_thinning_recipe_filters(resnet_cifar_model, zeros_mask_dict)

    if len(thinning_recipe.modules) > 0 or len(thinning_recipe.parameters)>0:
        # Stash the recipe, so that it will be serialized together with the model
        resnet_cifar_model.thinning_recipe = thinning_recipe
        # Now actually remove the filters, chaneels and make the weight tensors smaller
        execute_thinning_recipe(resnet_cifar_model, zeros_mask_dict, thinning_recipe)
        msglogger.info("Created, applied and saved a filter-thinning recipe")
    return resnet_cifar_model


def resnet_cifar_create_thinning_recipe_filters(resnet_cifar_model, zeros_mask_dict):
    """Remove filters from ResNet-Cifar

    Caveats: (1) supports only ResNet50-Cifar; (2) only module.layerX.Y.conv1.weight
    """
    msglogger.info("Invoking resnet_cifar_remove_filters")

    layers = {thin_name : m for thin_name, m in resnet_cifar_model.named_modules()}
    thinning_recipe = ThinningRecipe(modules={}, parameters={})

    for name, p_thin in resnet_cifar_model.named_parameters():
        if p_thin.dim() != 4:
            continue

        # Find the number of filters, in this weights tensor, that are not 100% sparse_model
        filter_view = p_thin.view(p_thin.size(0), -1)
        num_filters = filter_view.size()[0]
        nonzero_filters = torch.nonzero(filter_view.abs().sum(dim=1))

        # If there are zero-filters in this tensor then...
        if num_filters > len(nonzero_filters):
            msglogger.info("In tensor %s found %d/%d zero filters", name,
                           num_filters - len(nonzero_filters), num_filters)

            # Update the number of outgoing channels (OFMs) in the convolutional layer
            layer_name = name[:-len('weights')]
            assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
            thinning_recipe.modules[layer_name] = {}
            thinning_recipe.modules[layer_name]['out_channels'] = len(nonzero_filters)

            # Select only the non-zero filters
            indices = nonzero_filters.data.squeeze()
            thinning_recipe.parameters[name] = (0, indices)

            assert layer_name in conv_connectivity
            followers = conv_connectivity[layer_name] if isinstance(conv_connectivity[layer_name], list) else [conv_connectivity[layer_name]]
            for follower in followers:
                # For each of the convolutional layers that follow, we have to reduce the number of input channels.
                conv_directive = thinning_recipe.modules.get(follower, {})
                conv_directive['in_channels'] = len(nonzero_filters)
                thinning_recipe.modules[follower] = conv_directive

                # Now remove channels from the weights tensor of the follower conv
                #follower_param = distiller.model_find_param(resnet_cifar_model, follower+'.weight')
                thinning_recipe.parameters[follower+'.weight'] = (1, indices)

            # Now handle the BatchNormalization layer that follows the convolution
            bn_name = layer_name.replace('conv', 'bn')

            # Thinning of the BN layer that follows the convolution
            bn_thinning(thinning_recipe, layers, bn_name, len_thin_features=len(nonzero_filters), thin_features=indices)
    return thinning_recipe


class ResnetCifarChannelRemover(ScheduledTrainingPolicy):
    """A policy which applies a network thinning function"""
    def __init__(self, thinning_func_str):
        self.thinning_func = globals()[thinning_func_str]

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        self.thinning_func(model, zeros_mask_dict)


class ResnetCifarFilterRemover(ScheduledTrainingPolicy):
    """A policy which applies a network thinning function"""
    def __init__(self, thinning_func_str):
        self.thinning_func = globals()[thinning_func_str]
        self.done = False

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict):
        # We hook onto the on_minibatch_begin because we want to run after the pruner which sparsified
        # the tensors.  Pruners configure their pruning mask in on_epoch_begin, but apply the mask
        # only in on_minibatch_begin
        if not self.done:
            # We want to execute the thinning function only once, not every invocation of on_minibatch_begin
            self.thinning_func(model, zeros_mask_dict)
            self.done = True


def execute_thinning_recipe(model, zeros_mask_dict, recipe):
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
                setattr(layers[layer_name], attr,
                        torch.index_select(getattr(layers[layer_name], attr),
                                           dim=val[0], index=val[1]))
            else:
                setattr(layers[layer_name], attr, val)

    for param_name, info in recipe.parameters.items():
        param = distiller.model_find_param(model, param_name)
        param.data = torch.index_select(param.data, dim=info[0], index=info[1])
        mask = zeros_mask_dict[param_name].mask
        if (mask is not None) and (zeros_mask_dict[param_name].mask.size(dim=info[0]) != len(info[1])):
            zeros_mask_dict[param_name].mask = torch.index_select(mask, dim=info[0], index=info[1])

    model.thinning_recipe = recipe
    return model
