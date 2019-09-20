#
# Copyright (c) 2019 Intel Corporation
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
import numpy as np
from functools import partial


__all__ = ["kernels_lp_norm", "channels_lp_norm", "filters_lp_norm",
           "rows_lp_norm", "cols_lp_norm",
           "l1_norm", "l2_norm", "max_norm"]


class NamedFunction:
    def __init__(self, f, name):
        self.f = f
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.name


""" Norm (magnitude) functions.

These functions are named because it's convenient to refer to them when logging.
"""


def _max_norm(t, dim=1):
    maxv, _ = t.abs().max(dim=dim)
    return maxv


l1_norm = NamedFunction(partial(torch.norm, p=1, dim=1), "L1")
l2_norm = NamedFunction(partial(torch.norm, p=2, dim=1), "L2")
max_norm = NamedFunction(_max_norm, "Max")


def _norm_fn(p):
    norm_fn = p
    if not callable(p):
        norm_fn = partial(torch.norm, p=p, dim=1)
    return norm_fn


def kernels_lp_norm(param, p=1, length_normalized=False):
    """Compute the p-norm of 2D kernels of 4D parameter tensors.

    Assume 4D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation; or a callable that computes a normal
        length_normalized: if True then normalize the norm.  I.e.
        :: norm = channel_norm / num_elements_in_channel

    Returns:
        1D tensor with lp-norms of the channels (len = num_channels)
    """
    assert param.dim() == 4, "param has invalid dimensions"
    norm_fn = _norm_fn(p)
    with torch.no_grad():
        norm = norm_fn(param.view(-1, param.size(2) * param.size(3)))
        if length_normalized:
            norm = norm / np.prod(param.shape[2:])
        return norm


def channels_lp_norm(param, p=1, length_normalized=False):
    """Compute the p-norm of 3D channels of 4D parameter tensors.

    Assume 4D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation
        length_normalized: if True then normalize the norm.  I.e.
        :: norm = channel_norm / num_elements_in_channel

    Returns:
        1D tensor with lp-norms of the channels (len = num_channels)
    """
    assert param.dim() == 4, "param has invalid dimensions"
    norm_fn = _norm_fn(p)
    with torch.no_grad():
        param = param.transpose(0, 1).contiguous()
        norm = norm_fn(param.view(param.size(0), -1))
        if length_normalized:
            norm = norm / np.prod(param.shape[1:])
        return norm


def filters_lp_norm(param, p=1, length_normalized=False):
    """Compute the p-norm of 3D filters of 4D parameter tensors.

    Assume 4D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation
        length_normalized: if True then normalize the norm.  I.e.
        :: norm = filter_norm / num_elements_in_filter

    Returns:
        1D tensor with lp-norms of the filters (len = num_filters)
    """
    assert param.dim() == 4, "param has invalid dimensions"
    norm_fn = _norm_fn(p)
    with torch.no_grad():
        norm = norm_fn(param.view(param.size(0), -1))
        if length_normalized:
            norm = norm / np.prod(param.shape[1:])
        return norm


def matrix_lp_norm(param, p=1, length_normalized=False, dim=1):
    assert param.dim() == 2, "param has invalid dimensions"
    norm_fn = _norm_fn(p)
    with torch.no_grad():
        norm = norm_fn(param, dim=dim)
        if length_normalized:
            norm = norm / np.prod(param.shape(abs(dim-1)))
        return norm


def rows_lp_norm(param, p, length_normalized=False):
    return matrix_lp_norm(param, p, length_normalized, dim=1)


def cols_lp_norm(param, p, length_normalized=False):
    return matrix_lp_norm(param, p, length_normalized, dim=0)