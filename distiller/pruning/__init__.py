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

"""
:mod:`distiller.pruning` is a package implementing various pruning algorithms.
"""

from .magnitude_pruner import MagnitudeParameterPruner
from .automated_gradual_pruner import AutomatedGradualPruner, \
                                      L1RankedStructureParameterPruner_AGP, \
                                      L2RankedStructureParameterPruner_AGP, \
                                      ActivationAPoZRankedFilterPruner_AGP, \
                                      ActivationMeanRankedFilterPruner_AGP, \
                                      GradientRankedFilterPruner_AGP, \
                                      RandomRankedFilterPruner_AGP, \
                                      BernoulliFilterPruner_AGP
from .level_pruner import SparsityLevelParameterPruner
from .sensitivity_pruner import SensitivityPruner
from .splicing_pruner import SplicingPruner
from .structure_pruner import StructureParameterPruner
from .ranked_structures_pruner import L1RankedStructureParameterPruner, \
                                      L2RankedStructureParameterPruner, \
                                      ActivationAPoZRankedFilterPruner, \
                                      ActivationMeanRankedFilterPruner, \
                                      GradientRankedFilterPruner,       \
                                      RandomRankedFilterPruner,         \
                                      RandomLevelStructureParameterPruner, \
                                      BernoulliFilterPruner,            \
                                      FMReconstructionChannelPruner
from .baidu_rnn_pruner import BaiduRNNPruner
from .greedy_filter_pruning import greedy_pruner
import torch

del magnitude_pruner
del automated_gradual_pruner
del level_pruner
del sensitivity_pruner
del structure_pruner
del ranked_structures_pruner


def mask_tensor(tensor, mask, inplace=True):
    """Mask the provided tensor

    Args:
        tensor - the torch-tensor to mask
        mask - binary coefficient-masking tensor.  Has the same type and shape as `tensor`
    Returns:
        tensor = tensor * mask  ;where * is the element-wise multiplication operator
    """
    assert tensor.type() == mask.type()
    assert tensor.shape == mask.shape
    if mask is not None:
        return tensor.data.mul_(mask) if inplace else tensor.data.mul(mask)
    return tensor


def create_mask_threshold_criterion(tensor, threshold):
    """Create a tensor mask using a threshold criterion.

    All values smaller or equal to the threshold will be masked-away.
    Granularity: Element-wise
    Args:
        tensor - the tensor to threshold.
        threshold - a floating-point threshold value.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    with torch.no_grad():
        mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
        return mask


def create_mask_level_criterion(tensor, desired_sparsity):
    """Create a tensor mask using a level criterion.

    A specified fraction of the input tensor will be masked.  The tensor coefficients
    are first sorted by their L1-norm (absolute value), and then the lower `desired_sparsity`
    coefficients are masked.
    Granularity: Element-wise

    WARNING: due to the implementation details (speed over correctness), this will perform
    incorrectly if "too many" of the coefficients have the same value. E.g. this will fail:
        a = torch.ones(3, 64, 32, 32)
        mask = distiller.create_mask_level_criterion(a, desired_sparsity=0.3)
        assert math.isclose(distiller.sparsity(mask), 0.3)

    Args:
        tensor - the tensor to mask.
        desired_sparsity - a floating-point value in the range (0..1) specifying what
            percentage of the tensor will be masked.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    with torch.no_grad():
        # partial sort
        bottomk, _ = torch.topk(tensor.abs().view(-1),
                                int(desired_sparsity * tensor.numel()),
                                largest=False,
                                sorted=True)
        threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
        mask = create_mask_threshold_criterion(tensor, threshold)
        return mask


def create_mask_sensitivity_criterion(tensor, sensitivity):
    """Create a tensor mask using a sensitivity criterion.

    Mask an input tensor based on the variance of the distribution of the tensor coefficients.
    Coefficients in the distribution's specified band around the mean will be masked (symmetrically).
    Granularity: Element-wise
    Args:
        tensor - the tensor to mask.
        sensitivity - a floating-point value specifying the sensitivity.  This is a simple
            multiplier of the standard-deviation.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    if not hasattr(tensor, 'stddev'):
        tensor.stddev = torch.std(tensor).item()
    with torch.no_grad():
        threshold = tensor.stddev * sensitivity
        mask = create_mask_threshold_criterion(tensor, threshold)
        return mask
