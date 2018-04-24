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
import logging

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

        for k, v in bits_overrides.items():
            qbits = QBits(acts=v.get('acts', self.default_qbits.acts), wts=v.get('wts', self.default_qbits.wts))
            bits_overrides[k] = qbits

        # Prepare explicit mapping from each layer to QBits based on default + overrides
        if bits_overrides:
            regex_str = 
            keys_list = list(bits_overrides.keys())
            for pattern in keys_list:
                regex_str += '(^{0}$)|'.format(pattern)
            regex_str = regex_str[-1]   # Remove trailing '|'
            regex = re.compile(regex_str)

            self.layer_qbits_map = {}
            for layer_full_name, _ in model.named_modules():
                m = regex.match(layer_full_name)
                if m:
                    group_idx = 0
                    groups = m.groups()
                    while groups[group_idx] is None:
                        group_idx += 1
                    self.layer_qbits_map[layer_full_name] = bits_overrides[keys_list[group_idx]]
                else:
                    self.layer_qbits_map[layer_full_name] = self.default_qbits
        else:
            self.layer_qbits_map = {layer_full_name: self.default_qbits for layer_full_name, _ in model.named_modules()}

        self.replacement_factory = {}

    def prepare_model(self):
        msglogger.info('Preparing model for quantization')
        self._pre_process_container(self.model)

    def _pre_process_container(self, container, prefix=):
        # Iterate through model, insert quantization functions as appropriate
        for name, module in container.named_children():
            full_name = prefix + name
            try:
                new_module = self.replacement_factory[type(module)](module, full_name, self.layer_qbits_map)
                msglogger.debug('Module {0}: Replacing \n{1} with \n{2}'.format(full_name, module, new_module))
                container._modules[name] = new_module
            except KeyError:
                # For container we call recursively
                if len(module._modules) > 0:
                    self._pre_process_container(module, full_name + '.')
