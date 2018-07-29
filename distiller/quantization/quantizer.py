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

from collections import namedtuple, OrderedDict
import re
import copy
import logging
import torch
import torch.nn as nn
import distiller

msglogger = logging.getLogger()

QBits = namedtuple('QBits', ['acts', 'wts'])

FP_BKP_PREFIX = 'float_'


def has_bias(module):
    return hasattr(module, 'bias') and module.bias is not None


def hack_float_backup_parameter(module, name):
    try:
        data = dict(module.named_parameters())[name].data
    except KeyError:
        raise ValueError('Module has no Parameter named ' + name)
    module.register_parameter(FP_BKP_PREFIX + name, nn.Parameter(data))
    delattr(module, name)
    module.register_buffer(name, torch.zeros_like(data))


class _ParamToQuant(object):
    def __init__(self, module, module_name, fp_attr_name, q_attr_name, num_bits):
        self.module = module
        self.module_name = module_name
        self.fp_attr_name = fp_attr_name
        self.q_attr_name = q_attr_name
        self.num_bits = num_bits


class Quantizer(object):
    r"""
    Base class for quantizers.

    Args:
        model (torch.nn.Module): The model to be quantized
        optimizer (torch.optim.Optimizer): An optimizer instance, required in cases where the quantizer is going
            to perform changes to existing model parameters and/or add new ones.
            Specifically, when train_with_fp_copy is True, this cannot be None.
        bits_activations/weights (int): Default number of bits to use when quantizing each tensor type.
            Value of None means do not quantize.
        bits_overrides (OrderedDict): Dictionary mapping regular expressions of layer name patterns to dictionary with
            values for 'acts' and/or 'wts' to override the default values.
            OrderedDict is used to enable handling of overlapping name patterns. So, for example, one could define
            certain override parameters for a group of layers, e.g. 'conv*', but also define different parameters for
            specific layers in that group, e.g. 'conv1'.
            The patterns are evaluated eagerly - the first match wins. Therefore, the more specific patterns must
            come before the broad patterns.
        quantize_bias (bool): Flag indicating whether to quantize bias (w. same number of bits as weights) or not.
        train_with_fp_copy (bool): If true, will modify layers with weights to keep both a quantized and
            floating-point copy, such that the following flow occurs in each training iteration:
            1. q_weights = quantize(fp_weights)
            2. Forward through network using q_weights
            3. In back-prop:
                3.1 Gradients calculated with respect to q_weights
                3.2 We also back-prop through the 'quantize' operation from step 1
            4. Update fp_weights with gradients calculated in step 3.2
    """
    def __init__(self, model, optimizer=None, bits_activations=None, bits_weights=None, bits_overrides=OrderedDict(),
                 quantize_bias=False, train_with_fp_copy=False):
        if not isinstance(bits_overrides, OrderedDict):
            raise TypeError('bits_overrides must be an instance of collections.OrderedDict')
        if train_with_fp_copy and optimizer is None:
            raise ValueError('optimizer cannot be None when train_with_fp_copy is True')

        self.default_qbits = QBits(acts=bits_activations, wts=bits_weights)
        self.quantize_bias = quantize_bias

        self.model = model
        self.optimizer = optimizer

        # Stash some quantizer data in the model so we can re-apply the quantizer on a resuming model
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_weights': bits_weights,
                                                    'bits_overrides': copy.deepcopy(bits_overrides),
                                                    'quantize_bias': quantize_bias}}

        for k, v in bits_overrides.items():
            qbits = QBits(acts=v.get('acts', self.default_qbits.acts), wts=v.get('wts', self.default_qbits.wts))
            bits_overrides[k] = qbits

        # Prepare explicit mapping from each layer to QBits based on default + overrides
        patterns = []
        regex = None
        if bits_overrides:
            patterns = list(bits_overrides.keys())
            regex_str = '|'.join(['(^{0}$)'.format(pattern) for pattern in patterns])
            regex = re.compile(regex_str)

        self.module_qbits_map = {}
        for module_full_name, module in model.named_modules():
            qbits = self.default_qbits
            if regex:
                # Need to account for scenario where model is parallelized with DataParallel, which wraps the original
                # module with a wrapper module called 'module' :)
                name_to_match = module_full_name.replace('module.', '', 1)
                m = regex.match(name_to_match)
                if m:
                    group_idx = 0
                    groups = m.groups()
                    while groups[group_idx] is None:
                        group_idx += 1
                    qbits = bits_overrides[patterns[group_idx]]
            self._add_qbits_entry(module_full_name, type(module), qbits)

        # Mapping from module type to function generating a replacement module suited for quantization
        # To be populated by child classes
        self.replacement_factory = {}
        # Pointer to parameters quantization function, triggered during training process
        # To be populated by child classes
        self.param_quantization_fn = None

        self.train_with_fp_copy = train_with_fp_copy
        self.params_to_quantize = []

    def _add_qbits_entry(self, module_name, module_type, qbits):
        if module_type not in [nn.Conv2d, nn.Linear]:
            # For now we support weights quantization only for Conv and FC layers (so, for example, we don't
            # support quantization of batch norm scale parameters)
            qbits = QBits(acts=qbits.acts, wts=None)
        self.module_qbits_map[module_name] = qbits

    def prepare_model(self):
        r"""
        Iterates over the model and replaces modules with their quantized counterparts as defined by
        self.replacement_factory
        """
        msglogger.info('Preparing model for quantization using {0}'.format(self.__class__.__name__))
        self._pre_process_container(self.model)

        for module_name, module in self.model.named_modules():
            qbits = self.module_qbits_map[module_name]
            if qbits.wts is None:
                continue

            curr_parameters = dict(module.named_parameters())
            for param_name, param in curr_parameters.items():
                if param_name.endswith('bias') and not self.quantize_bias:
                    continue
                fp_attr_name = param_name
                if self.train_with_fp_copy:
                    hack_float_backup_parameter(module, param_name)
                    fp_attr_name = FP_BKP_PREFIX + param_name
                self.params_to_quantize.append(_ParamToQuant(module, module_name, fp_attr_name, param_name, qbits.wts))

                param_full_name = '.'.join([module_name, param_name])
                msglogger.info(
                    "Parameter '{0}' will be quantized to {1} bits".format(param_full_name, qbits.wts))

        # If an optimizer was passed, assume we need to update it
        if self.optimizer:
            optimizer_type = type(self.optimizer)
            new_optimizer = optimizer_type(self._get_updated_optimizer_params_groups(), **self.optimizer.defaults)
            self.optimizer.__setstate__({'param_groups': new_optimizer.param_groups})

        msglogger.info('Quantized model:\n\n{0}\n'.format(self.model))

    def _pre_process_container(self, container, prefix=''):
        # Iterate through model, insert quantization functions as appropriate
        for name, module in container.named_children():
            full_name = prefix + name
            current_qbits = self.module_qbits_map[full_name]
            if current_qbits.acts is None and current_qbits.wts is None:
                continue
            try:
                new_module = self.replacement_factory[type(module)](module, full_name, self.module_qbits_map)
                msglogger.debug('Module {0}: Replacing \n{1} with \n{2}'.format(full_name, module, new_module))
                setattr(container, name, new_module)

                # If a "leaf" module was replaced by a container, add the new layers to the QBits mapping
                if not distiller.has_children(module) and distiller.has_children(new_module):
                    for sub_module_name, sub_module in new_module.named_modules():
                        self._add_qbits_entry(full_name + '.' + sub_module_name, type(sub_module), current_qbits)
                    self.module_qbits_map[full_name] = QBits(acts=current_qbits.acts, wts=None)
            except KeyError:
                pass

            if distiller.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

    def _get_updated_optimizer_params_groups(self):
        """
        Returns a list of model parameter groups and optimizer hyper-parameter overrides,
        as expected by the __init__ function of torch.optim.Optimizer.
        This is called after all model changes were made in prepare_model, in case an Optimizer instance was
        passed to __init__.

        Subclasses which add parameters to the model should override as needed.

        :return: List of parameter groups
        """
        # Default implementation - just return all model parameters as one group
        return [{'params': self.model.parameters()}]

    def quantize_params(self):
        """
        Quantize all parameters using the parameters using self.param_quantization_fn (using the defined number
        of bits for each parameter)
        """
        for ptq in self.params_to_quantize:
            q_param = self.param_quantization_fn(getattr(ptq.module, ptq.fp_attr_name), ptq.num_bits)
            if self.train_with_fp_copy:
                setattr(ptq.module, ptq.q_attr_name, q_param)
            else:
                getattr(ptq.module, ptq.q_attr_name).data = q_param.data
