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
DistributionRegularizer:
    Regularizes weights to a target distribution, calculating a Soft-Histogram on the weights and then
    KLDivLoss between the soft-probability distrbution and the target distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .regularizer import _Regularizer
from collections import OrderedDict
import distiller


class DistributionRegularizer(_Regularizer):
    """
    Creates a distribution regularizer.
    """
    def __init__(self, name, model, reg_regims, distribution_kwargs_sched, target_distribution_fn=None, n_bins=1024):
        """
        Args:
            name (str): name of regularizer
            model (nn.Module): the model to regularize.
            reg_regims (dict[str,tuple[float, str]]): regularization regiment.  A dictionary of
                        reg_regims[<param-name>] = [ lambda, structure-type]
            distribution_kwargs_sched (dict[int, dict]): scheduling for the temperature parameter of the target distribution.
            target_distribution_fn (str or callable): generates a target distribution for a given weight tensor.
              assumes `target_distribution_fn(weight: torch.Tensor, n_bins: int, *args)->ProbDist`.
              if is str: take from a known implemented target distribution.
            n_bins (int): histogram number of bins
        """
        super(DistributionRegularizer, self).__init__(name, model, reg_regims, None)
        if isinstance(target_distribution_fn, str):
            target_distribution_fn = _STR_TO_F[target_distribution_fn]
        self.target_distribution_fn = target_distribution_fn
        self.distribution_args_sched = distribution_kwargs_sched
        self.n_bins = n_bins

        self.target_distributions = {}  # type: dict[str, ProbDist]

    def on_epoch_begin(self, epoch):
        weights_dict = OrderedDict(self.model.named_parameters())
        args = self.distribution_args_sched[epoch]
        self.target_distributions = {p_name: self.target_distribution_fn(weights_dict[p_name], self.n_bins, *args)
                                     for p_name in self.reg_regims}

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict=None):
        target_probdist = self.target_distributions[param_name]
        current_probdist = soft_probdist(param, target_probdist.bins)
        return F.kl_div(current_probdist.distribution, target_probdist.distribution)

    def threshold(self, param, param_name, zeros_mask_dict):
        raise NotImplementedError


class SoftProbdist(nn.Module):
    def __init__(self, histogram_bins, scale=None):
        super(SoftProbdist, self).__init__()
        self.bins = histogram_bins
        self.scale = scale or self.bins[1] - self.bins[0]

    def forward(self, tensor):
        """
        Calculates a differentiable probability density function approximation of a tensor.
        Args:
            tensor (torch.Tensor): the input variable
        """
        return soft_probdist(tensor, self.bins, self.scale)


def soft_probdist(tensor, bins, scale=None):
    if scale is None:
        scale = bins[1] - bins[0]
    tensor, bins = tensor.view(1, -1), bins.unsqueeze(1)
    x = tensor - bins
    x = torch.exp(-((x / (0.5 * scale)) ** 2))
    soft_hist = x.sum(dim=1)
    probdist = soft_hist / soft_hist.sum()
    return ProbDist(probdist, bins)


class ProbDist:
    def __init__(self, distribution, bins):
        self.distribution = distribution  # type: torch.Tensor
        self.bins = bins  # type: torch.Tensor


def laplace_linear_mixture(tensor, n_bins, temperature=1.0, n_centers=255):
    """
    Generates a target linear mixture of Laplace distributions out of tensor of values.
    Args:
        tensor (torch.Tensor): values
        n_bins (int): number of bins in the histogram
        temperature (float): controls the sharpness of the Laplace Distributions
        n_centers (int): number of laplace distributions
    """
    with torch.no_grad():
        m, M = tensor.min(), tensor.max()
        scale_bins, scale_centers = (M-m) / (n_bins-1), (M-m) / (n_centers-1)
        bins = torch.arange(-n_bins//2, n_bins//2) * scale_bins
        centers = torch.arange(-n_centers//2, n_centers//2) * scale_centers


_STR_TO_F = {
    'LLM': laplace_linear_mixture
}


