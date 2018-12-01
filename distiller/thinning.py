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
from collections import namedtuple
import torch
from .policy import ScheduledTrainingPolicy
import distiller
from distiller import normalize_module_name, denormalize_module_name
from apputils import SummaryGraph
from models import create_model
msglogger = logging.getLogger(__name__)

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
           'StructureRemover',
           'ChannelRemover', 'remove_channels',
           'FilterRemover',  'remove_filters',
           'find_nonzero_channels', 'find_nonzero_channels_list',
           'execute_thinning_recipes_list']


def create_graph(dataset, arch):
    if dataset == 'imagenet':
        dummy_input = torch.randn((1, 3, 224, 224), requires_grad=False)
    elif dataset == 'cifar10':
        dummy_input = torch.randn((1, 3, 32, 32))
    assert dummy_input is not None, "Unsupported dataset ({}) - aborting draw operation".format(dataset)

    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None
    return SummaryGraph(model, dummy_input.cuda())


def param_name_2_layer_name(param_name):
    return param_name[:-len('weights')]


def directives_equal(d1, d2):
    """Test if two directives are equal"""
    if len(d1) != len(d2):
        return False
    if len(d1) == 2:
        return d1[0] == d2[0] and torch.equal(d1[1], d2[1])
    if len(d1) == 4:
        e = all(d1[i] == d2[i] for i in (0, 2, 3)) and torch.equal(d1[1], d2[1])
        msglogger.info("{}: \n{}\n{}".format(e, d1, d2))
        return e
    assert ValueError("Unsupported directive length")


def append_param_directive(thinning_recipe, param_name, directive):
    param_directives = thinning_recipe.parameters.get(param_name, [])
    for d in param_directives:
        # Duplicate parameter directives are rooted out because they can create erronous conditions.
        # For example, if the first directive changes the change of the parameter, a second
        # directive will cause an exception.
        if directives_equal(d, directive):
            return
    msglogger.debug("\t[recipe] param_directive for {} = {}".format(param_name, directive))
    param_directives.append(directive)
    thinning_recipe.parameters[param_name] = param_directives


def append_module_directive(model, thinning_recipe, module_name, key, val):
    msglogger.debug("\t[recipe] setting {}.{} = {}".format(module_name, key, val))
    module_name = denormalize_module_name(model, module_name)
    mod_directive = thinning_recipe.modules.get(module_name, {})
    mod_directive[key] = val
    thinning_recipe.modules[module_name] = mod_directive


def append_bn_thinning_directive(thinning_recipe, layers, bn_name, len_thin_features, thin_features):
    """Adjust the sizes of the parameters of a BatchNormalization layer
    This function is invoked after the Convolution layer preceeding a BN layer has
    changed dimensions (filters or channels were removed), and the BN layer also
    requires updating as a result.
    """
    bn_module = layers[bn_name]
    assert isinstance(bn_module, torch.nn.modules.batchnorm.BatchNorm2d)
    msglogger.debug("\t[recipe] bn_thinning {}".format(bn_name))

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


def remove_channels(model, zeros_mask_dict, arch, dataset, optimizer):
    sgraph = create_graph(dataset, arch)
    thinning_recipe = create_thinning_recipe_channels(sgraph, model, zeros_mask_dict)
    apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe, optimizer)
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

    if num_channels > nonzero_channels.nelement():
        msglogger.info("In tensor %s found %d/%d zero channels", param_name,
                       num_channels - nonzero_channels.nelement(), num_channels)

    return nonzero_channels


def find_nonzero_channels_list(param, param_name):
    nnz_channels = find_nonzero_channels(param, param_name)
    nnz_channels = nnz_channels.view(nnz_channels.numel())
    return nnz_channels.cpu().numpy().tolist()


def apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe, optimizer):
    if len(thinning_recipe.modules) > 0 or len(thinning_recipe.parameters) > 0:
        # Now actually remove the filters, chaneels and make the weight tensors smaller
        execute_thinning_recipe(model, zeros_mask_dict, thinning_recipe, optimizer)

        # Stash the recipe, so that it will be serialized together with the model
        if hasattr(model, 'thinning_recipes'):
            # Add the new recipe to the existing recipes.  They will be applied by order.
            model.thinning_recipes.append(thinning_recipe)
        else:
            model.thinning_recipes = [thinning_recipe]
        msglogger.info("Created, applied and saved a thinning recipe")
    else:
        msglogger.error("Failed to create a thinning recipe")


def remove_filters(model, zeros_mask_dict, arch, dataset, optimizer):
    sgraph = create_graph(dataset, arch)
    thinning_recipe = create_thinning_recipe_filters(sgraph, model, zeros_mask_dict)
    apply_and_save_recipe(model, zeros_mask_dict, thinning_recipe, optimizer)
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
    layers = {mod_name: m for mod_name, m in model.named_modules()}

    # Traverse all of the model's parameters, search for zero-channels, and
    # create a thinning recipe that descibes the required changes to the model.
    for param_name, param in model.named_parameters():
        # We are only interested in 4D weights (of Convolution layers)
        if param.dim() != 4:
            continue

        num_channels = param.size(1)
        nonzero_channels = find_nonzero_channels(param, param_name)
        num_nnz_channels = nonzero_channels.nelement()
        if num_nnz_channels == 0:
            raise ValueError("Trying to set zero channels for parameter %s is not allowed" % param_name)
        # If there are non-zero channels in this tensor then continue to next tensor
        if num_channels <= num_nnz_channels:
            continue

        # We are removing channels, so update the number of incoming channels (IFMs)
        # in the convolutional layer
        layer_name = param_name_2_layer_name(param_name)
        assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
        append_module_directive(model, thinning_recipe, layer_name, key='in_channels', val=num_nnz_channels)

        # Select only the non-zero filters
        indices = nonzero_channels.data.squeeze()
        append_param_directive(thinning_recipe, param_name, (1, indices))

        # Find all instances of Convolution layers that immediately preceed this layer
        predecessors = sgraph.predecessors_f(normalize_module_name(layer_name), ['Conv'])
        # Convert the layers names to PyTorch's convoluted naming scheme (when DataParallel is used)
        predecessors = [normalize_module_name(predecessor) for predecessor in predecessors]
        if len(predecessors) == 0:
            msglogger.info("Could not find predecessors for name={} normal={} {}".format(
                           layer_name, normalize_module_name(layer_name), denormalize_module_name(model, layer_name)))
        for predecessor in predecessors:
            # For each of the convolutional layers that preceed, we have to reduce the number of output channels.
            append_module_directive(model, thinning_recipe, predecessor, key='out_channels', val=num_nnz_channels)

            # Now remove channels from the weights tensor of the predecessor conv
            append_param_directive(thinning_recipe, denormalize_module_name(model, predecessor)+'.weight', (0, indices))

            if layers[denormalize_module_name(model, predecessor)].bias is not None:
                # This convolution has bias coefficients
                append_param_directive(thinning_recipe, denormalize_module_name(model, predecessor)+'.bias', (0, indices))

        # Now handle the BatchNormalization layer that follows the convolution
        bn_layers = sgraph.predecessors_f(normalize_module_name(layer_name), ['BatchNormalization'])
        if len(bn_layers) > 0:
            # if len(bn_layers) != 1:
            #     raise RuntimeError("{} should have exactly one BN predecessors, but has {}".format(layer_name, len(bn_layers)))
            for bn_layer in bn_layers:
                # Thinning of the BN layer that follows the convolution
                bn_layer_name = denormalize_module_name(model, bn_layer)
                msglogger.debug("[recipe] {}: predecessor BN module = {}".format(layer_name, bn_layer_name))
                append_bn_thinning_directive(thinning_recipe, layers, bn_layer_name,
                                             len_thin_features=num_nnz_channels, thin_features=indices)

    msglogger.debug(thinning_recipe)
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
    msglogger.debug(sgraph.ops.keys())

    thinning_recipe = ThinningRecipe(modules={}, parameters={})
    layers = {mod_name: m for mod_name, m in model.named_modules()}

    for param_name, param in model.named_parameters():
        # We are only interested in 4D weights
        if param.dim() != 4:
            continue

        # Find the number of zero-valued filters in this weights tensor
        filter_view = param.view(param.size(0), -1)
        num_filters = filter_view.size()[0]
        nonzero_filters = torch.nonzero(filter_view.abs().sum(dim=1))
        num_nnz_filters = nonzero_filters.nelement()
        if num_nnz_filters == 0:
            raise ValueError("Trying to set zero filters for parameter %s is not allowed" % param_name)
        # If there are non-zero filters in this tensor then continue to next tensor
        if num_filters <= num_nnz_filters:
            msglogger.debug("Skipping {} shape={}".format(param_name_2_layer_name(param_name), param.shape))
            continue

        msglogger.info("In tensor %s found %d/%d zero filters", param_name,
                       num_filters - num_nnz_filters, num_filters)

        # We are removing filters, so update the number of outgoing channels (OFMs)
        # in the convolutional layer
        layer_name = param_name_2_layer_name(param_name)
        assert isinstance(layers[layer_name], torch.nn.modules.Conv2d)
        append_module_directive(model, thinning_recipe, layer_name, key='out_channels', val=num_nnz_filters)

        # Select only the non-zero filters
        indices = nonzero_filters.data.squeeze()
        append_param_directive(thinning_recipe, param_name, (0, indices))

        if layers[layer_name].bias is not None:
            # This convolution has bias coefficients
            append_param_directive(thinning_recipe, layer_name+'.bias', (0, indices))

        # Find all instances of Convolution or FC (GEMM) layers that immediately follow this layer
        msglogger.debug("{} => {}".format(layer_name, normalize_module_name(layer_name)))
        successors = sgraph.successors_f(normalize_module_name(layer_name), ['Conv', 'Gemm'])
        # Convert the layers names to PyTorch's convoluted naming scheme (when DataParallel is used)
        successors = [denormalize_module_name(model, successor) for successor in successors]
        for successor in successors:

            if isinstance(layers[successor], torch.nn.modules.Conv2d):
                # For each of the convolutional layers that follow, we have to reduce the number of input channels.
                append_module_directive(model, thinning_recipe, successor, key='in_channels', val=num_nnz_filters)

                # Now remove channels from the weights tensor of the successor conv
                append_param_directive(thinning_recipe, denormalize_module_name(model, successor)+'.weight', (1, indices))

            elif isinstance(layers[successor], torch.nn.modules.Linear):
                # If a Linear (Fully-Connected) layer follows, we need to update it's in_features member
                fm_size = layers[successor].in_features // layers[layer_name].out_channels
                in_features = fm_size * num_nnz_filters
                append_module_directive(model, thinning_recipe, successor, key='in_features', val=in_features)
                msglogger.debug("[recipe] Linear {}: fm_size = {}  layers[{}].out_channels={}".format(
                                successor, in_features, layer_name, layers[layer_name].out_channels))
                msglogger.debug("[recipe] {}: setting in_features = {}".format(successor, in_features))

                # Now remove channels from the weights tensor of the successor FC layer:
                # This is a bit tricky:
                fm_height = fm_width = int(math.sqrt(fm_size))
                view_4D = (layers[successor].out_features, layers[layer_name].out_channels, fm_height, fm_width)
                view_2D = (layers[successor].out_features, in_features)
                append_param_directive(thinning_recipe,
                                       denormalize_module_name(model, successor)+'.weight',
                                       (1, indices, view_4D, view_2D))

        # Now handle the BatchNormalization layer that follows the convolution
        bn_layers = sgraph.successors_f(normalize_module_name(layer_name), ['BatchNormalization'])
        if len(bn_layers) > 0:
            assert len(bn_layers) == 1
            # Thinning of the BN layer that follows the convolution
            bn_layer_name = denormalize_module_name(model, bn_layers[0])
            append_bn_thinning_directive(thinning_recipe, layers, bn_layer_name,
                                         len_thin_features=num_nnz_filters, thin_features=indices)
    return thinning_recipe


class StructureRemover(ScheduledTrainingPolicy):
    """A policy which applies a network thinning function"""
    def __init__(self, thinning_func_str, arch, dataset):
        self.thinning_func = globals()[thinning_func_str]
        self.arch = arch
        self.dataset = dataset
        self.done = False
        self.active_cb = "on_minibatch_begin"

    def __apply(self, model, zeros_mask_dict, optimizer):
        if not self.done:
            # We want to execute the thinning function only once, not every invocation of on_minibatch_begin
            self.thinning_func(model, zeros_mask_dict, self.arch, self.dataset, optimizer=optimizer)
            self.done = True

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, meta, optimizer):
        # We hook onto the on_minibatch_begin because we want to run after the pruner which sparsified
        # the tensors.  Pruners configure their pruning mask in on_epoch_begin, but apply the mask
        # only in on_minibatch_begin
        if self.active_cb != "on_minibatch_begin":
            return
        self.__apply(model, zeros_mask_dict, optimizer)

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.active_cb != "on_minibatch_end":
            return
        self.__apply(model, zeros_mask_dict, optimizer)

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        # The epoch has ended and we reset the 'done' flag, so that the FilterRemover instance can be reused
        self.done = False


# For backward-compatibility with some of the scripts, we assign aliases to StructureRemover
FilterRemover = StructureRemover
ChannelRemover = StructureRemover


def execute_thinning_recipes_list(model, zeros_mask_dict, recipe_list):
    # Invoke this function when you want to use a list of thinning recipes to convert a programmed model
    # to a thinned model. For example, this is invoked when loading a model from a checkpoint.
    for i, recipe in enumerate(recipe_list):
        msglogger.debug("Executing recipe %d:" % i)
        execute_thinning_recipe(model, zeros_mask_dict, recipe, optimizer=None, loaded_from_file=True)
    msglogger.info("Executed %d recipes" % len(recipe_list))


def optimizer_thinning(optimizer, param, dim, indices, new_shape=None):
    """Adjust the size of the SGD velocity-tracking tensors.

    The SGD momentum update (velocity) is dependent on the weights, and because during thinning we
    dynamically change the weights shapes, we need to make the apporpriate changes in the Optimizer,
    or disable the momentum.

    This function is brittle as it is tested on SGD only and relies on the internal representation of
    the SGD optimizer, which can change w/o notice.
    """
    if optimizer is None or not isinstance(optimizer, torch.optim.SGD):
        return False
    for group in optimizer.param_groups:
        momentum = group.get('momentum', 0)
        if momentum == 0:
            continue
        for p in group['params']:
            if id(p) != id(param):
                continue
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                param_state['momentum_buffer'] = torch.index_select(param_state['momentum_buffer'], dim, indices)
                if new_shape is not None:
                    msglogger.debug("optimizer_thinning: new shape {}".format(*new_shape))
                    param_state['momentum_buffer'] = param_state['momentum_buffer'].resize_(*new_shape)
                return True
    return False


def execute_thinning_recipe(model, zeros_mask_dict, recipe, optimizer, loaded_from_file=False):
    """Apply a thinning recipe to a model.
    This will remove filters and channels, as well as handle batch-normalization parameter
    adjustment, and thinning of weight tensors.
    """

    layers = {mod_name: m for mod_name, m in model.named_modules()}
    for layer_name, directives in recipe.modules.items():
        for attr, val in directives.items():
            if attr in ['running_mean', 'running_var']:
                running = getattr(layers[layer_name], attr)
                dim_to_trim = val[0]
                indices_to_select = val[1]
                # Check if we're trying to trim a parameter that is already "thin"
                if running.size(dim_to_trim) != indices_to_select.nelement():
                    msglogger.debug("[thinning] {}: setting {} to {}".
                                    format(layer_name, attr, indices_to_select.nelement()))
                    setattr(layers[layer_name], attr,
                            torch.index_select(running, dim=dim_to_trim, index=indices_to_select))
            else:
                msglogger.debug("[thinning] {}: setting {} to {}".format(layer_name, attr, val))
                setattr(layers[layer_name], attr, val)

    assert len(recipe.parameters) > 0

    for param_name, param_directives in recipe.parameters.items():
        msglogger.debug("{} : {}".format(param_name, param_directives))
        param = distiller.model_find_param(model, param_name)
        assert param is not None
        for directive in param_directives:
            dim = directive[0]
            indices = directive[1]
            len_indices = indices.nelement()
            if len(directive) == 4:  # TODO: this code is hard to follow
                msglogger.debug("{}-{}-{}: SHAPE = {}".format(param_name, param.shape, id(param), list(directive[2])))
                selection_view = param.view(*directive[2])
                # Check if we're trying to trim a parameter that is already "thin"
                if param.data.size(dim) != len_indices:
                    param.data = torch.index_select(selection_view, dim, indices)
                    if param.grad is not None:
                        # We also need to change the dimensions of the gradient tensor.
                        grad_selection_view = param.grad.resize_(*directive[2])
                        if grad_selection_view.size(dim) != len_indices:
                            param.grad = torch.index_select(grad_selection_view, dim, indices)
                            if optimizer_thinning(optimizer, param, dim, indices, directive[3]):
                                msglogger.debug("Updated [4D] velocity buffer for {} (dim={},size={},shape={})".
                                                format(param_name, dim, len_indices, directive[3]))

                param.data = param.view(*directive[3])
                if param.grad is not None:
                    param.grad = param.grad.resize_(*directive[3])
            else:
                if param.data.size(dim) != len_indices:
                    param.data = torch.index_select(param.data, dim, indices)
                    msglogger.debug("[thinning] changed param {} shape: {}".format(param_name, len_indices))
                # We also need to change the dimensions of the gradient tensor.
                # If have not done a backward-pass thus far, then the gradient will
                # not exist, and therefore won't need to be re-dimensioned.
                if param.grad is not None and param.grad.size(dim) != len_indices:
                    param.grad = torch.index_select(param.grad, dim, indices)
                    if optimizer_thinning(optimizer, param, dim, indices):
                        msglogger.debug("Updated velocity buffer %s" % param_name)

            if not loaded_from_file:
                # If the masks are loaded from a checkpoint file, then we don't need to change
                # their shape, because they are already correctly shaped
                mask = zeros_mask_dict[param_name].mask
                if mask is not None and (mask.size(dim) != len_indices):
                    zeros_mask_dict[param_name].mask = torch.index_select(mask, dim, indices)
