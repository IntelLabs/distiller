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

from collections import namedtuple
import re
import copy
import logging
import torch.nn as nn

msglogger = logging.getLogger()

QBits = namedtuple('QBits', ['acts', 'wts'])


class Quantizer(object):
    r"""
    Base class for quantizers

    Args:
        model (torch.nn.Module): The model to be quantized
        bits_activations/weights (int): Default number of bits to use when quantizing each tensor type.
            Value of None means do not quantize.
        bits_overrides (dict): Dictionary mapping regular expressions of layer name patterns to dictionary with
            values for 'acts' and/or 'wts' to override the default values.
    """
    def __init__(self, model, bits_activations=None, bits_weights=None, bits_overrides={}):
        self.default_qbits = QBits(acts=bits_activations, wts=bits_weights)

        self.model = model

        # Stash some quantizer data in the model so we can re-apply the quantizer on a resuming model
        self.model.quantizer_metadata = {'type': type(self), 'params': {'bits_activations': bits_activations,
                                                                        'bits_weights': bits_weights,
                                                                        'bits_overrides': copy.deepcopy(bits_overrides)}}

        for k, v in bits_overrides.items():
            qbits = QBits(acts=v.get('acts', self.default_qbits.acts), wts=v.get('wts', self.default_qbits.wts))
            bits_overrides[k] = qbits

        # Prepare explicit mapping from each layer to QBits based on default + overrides
        regex = None
        if bits_overrides:
            regex_str = ''
            keys_list = list(bits_overrides.keys())
            for pattern in keys_list:
                regex_str += '(^{0}$)|'.format(pattern)
            regex_str = regex_str[:-1]   # Remove trailing '|'
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
                    qbits = bits_overrides[keys_list[group_idx]]
            self._add_qbits_entry(module_full_name, type(module), qbits)

        # Mapping from module type to function generating a replacement module suited for quantization
        # To be populated by child classes
        self.replacement_factory = {}
        # Pointer to parameters quantization function, triggered during training process
        # To be populated by child classes
        self.param_quantization_fn = None

        # Mapping from parameter name to number of bits, used by quantize_param() function. Populated by prepare_model()
        self.quantizable_params = {}

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
        msglogger.info('Preparing model for quantization')
        self._pre_process_container(self.model)
        for module_name, module in self.model.named_modules():
            qbits = self.module_qbits_map[module_name]
            if qbits.wts is not None:
                for param_name, _ in module.named_parameters():
                    self.quantizable_params['.'.join([module_name, param_name])] = qbits.wts
        msglogger.info('Quantized model:')
        msglogger.info('')
        msglogger.info(self.model)
        msglogger.info('')

    def _pre_process_container(self, container, prefix=''):
        # Iterate through model, insert quantization functions as appropriate
        for name, module in container.named_children():
            full_name = prefix + name
            try:
                new_module = self.replacement_factory[type(module)](module, full_name, self.module_qbits_map)
                msglogger.debug('Module {0}: Replacing \n{1} with \n{2}'.format(full_name, module, new_module))
                container._modules[name] = new_module

                # If a "leaf" module was replaced by a container, add the new layers to the QBits mapping
                if len(module._modules) == 0 and len(new_module._modules) > 0:
                    current_qbits = self.module_qbits_map[full_name]
                    for sub_module_name, module in new_module.named_modules():
                        self._add_qbits_entry(full_name + '.' + sub_module_name, type(module), current_qbits)
                    self.module_qbits_map[full_name] = QBits(acts=current_qbits.acts, wts=None)
            except KeyError:
                pass

            if len(module._modules) > 0:
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

    def quantize_param(self, param_name, param_fp):
        r"""
        Quantize a parameter tensor according the number of bits set in the initialization
        """
        if not self.param_quantization_fn:
            return param_fp
        try:
            return self.param_quantization_fn(param_fp, self.quantizable_params[param_name])
        except KeyError:
            return param_fp
