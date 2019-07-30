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
import numpy as np
import torch
from torch.nn import functional as f
import distiller


msglogger = logging.getLogger()


def basic_featuremaps_caching_fwd_hook(module, input, output, intermediate_fms):
    """A trivial function to cache input/output feature-maps
    
    The input feature-maps are appended to a list of input-maps that are input to
    this module.  This list is provided by an external context.  A similar setup
    exists for output feature-maps.

    This function is invoked from the forward-hook of modules and can be called from
    various threads and the modules can exist on multiple GPUs.  Therefore, we use Python
    lists (on the CPU) to protect against race-conditions and synchronize the data.
    Using the CPU to store the lists also benefits from the larger CPU DRAM.
    """
    intermediate_fms['output_fms'][module.distiller_name].append(output)
    intermediate_fms['input_fms'][module.distiller_name].append(input[0])


def collect_intermediate_featuremap_samples(model, forward_fn, module_filter_fn, 
                                            fm_caching_fwd_hook=basic_featuremaps_caching_fwd_hook):
    """Collect pairs of input/output feature-maps.
    """
    from functools import partial

    def install_io_collectors(m, intermediate_fms):
        if module_filter_fn(m):
            intermediate_fms['output_fms'][m.distiller_name] = []
            intermediate_fms['input_fms'][m.distiller_name] = []
            hook_handles.append(m.register_forward_hook(partial(fm_caching_fwd_hook, 
                                                                intermediate_fms=intermediate_fms)))

    # Register to the forward hooks, then run the forward-pass and collect the data
    msglogger.warning("==> Collecting input/ouptput feature-map pairs")
    distiller.assign_layer_fq_names(model)
    hook_handles = []
    intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
    model.apply(partial(install_io_collectors, intermediate_fms=intermediate_fms))
    
    forward_fn()
    
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    # We now need to concatenate the list of feature-maps to torch tensors.
    msglogger.info("Concatenating FMs...")
    model.intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
    outputs = model.intermediate_fms['output_fms']
    inputs = model.intermediate_fms['input_fms']

    for (layer_name, X), Y in zip(intermediate_fms['input_fms'].items(), intermediate_fms['output_fms'].values()):                
        inputs[layer_name] = torch.cat(X, dim=0)
        outputs[layer_name] = torch.cat(Y, dim=0)

    msglogger.warning("<== Done.")
    del intermediate_fms 