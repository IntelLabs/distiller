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

import torch
import distiller


__all__ = ["mask_tensor"]


class _ParameterPruner(object):
    """Base class for all pruners.

    Arguments:
        name: pruner name is used mainly for debugging.
    """
    def __init__(self, name):
        self.name = name

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        raise NotImplementedError


def mask_tensor(tensor, mask):
    """Mask the provided tensor

    Args:
        tensor - the torch-tensor to mask
        mask - binary coefficient-masking tensor.  Has the same type and shape as `tensor`
    Returns:
        tensor = tensor * mask  ;where * is the element-wise multiplication operator
    """
    assert tensor.type() == mask.type()
    assert tensor.shape == mask.shape
    if mask:
        tensor.data.mul_(mask)
    return tensor
