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
from collections import OrderedDict, namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.axes import Axes
import os
import distiller

_EPS = 1e-6


class DistributionRegularizer(_Regularizer):
    """
    Creates a distribution regularizer.
    """
    def __init__(self, name, model, reg_regims, distribution_kwargs_sched, target_distribution_fn=None, n_bins=1024):
        """
        Args:
            name (str): name of regularizer
            model (nn.Module): the model to regularize.
            reg_regims (dict[str,float]): regularization regiment.  A dictionary of
                        reg_regims[<param-name>] = lambda
            distribution_kwargs_sched (dict[int, dict]): scheduling for the temperature parameter of the target distribution.
            target_distribution_fn (str or callable): generates a target distribution for a given weight tensor.
              assumes `target_distribution_fn(weight: torch.Tensor, n_bins: int, **kwargs)->ProbDist`.
              if is str: take from a known implemented target distribution.
            n_bins (int): histogram number of bins
        """
        super(DistributionRegularizer, self).__init__(name, model, reg_regims, None)
        if isinstance(target_distribution_fn, str):
            target_distribution_fn = _STR_TO_F[target_distribution_fn]
        self.target_distribution_fn = target_distribution_fn
        self.distribution_kwargs_sched = distribution_kwargs_sched
        self.n_bins = n_bins

        self.target_distributions = {}  # type: dict[str, ProbDist]

    def on_epoch_begin(self, epoch):
        weights_dict = OrderedDict(self.model.named_parameters())
        kwargs = self.distribution_kwargs_sched.get(epoch, None)
        if kwargs is None:
            return  # keep the previous distributions
        self.target_distributions = {p_name: self.target_distribution_fn(weights_dict[p_name], self.n_bins, **kwargs)
                                     for p_name in self.reg_regims}

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict=None):
        target_probdist = self.target_distributions.get(param_name, None)
        if target_probdist is None:
            return regularizer_loss  # No change in loss
        current_probdist = soft_probdist(param, target_probdist.bins)
        strength = self.reg_regims[param_name]
        input_probdist = current_probdist.distribution.unsqueeze(0)
        target_probdist = target_probdist.distribution.unsqueeze(0)
        current_loss = strength * param.numel() * stable_kldiv(input_probdist, target_probdist)
        regularizer_loss += current_loss
        return regularizer_loss

    def threshold(self, param, param_name, zeros_mask_dict):
        raise NotImplementedError


def stable_kldiv(input_prob, target_prob):
    """
    Numerically stable KL-Divergence.
    Args:
        input_prob (torch.Tensor): the input
        target_prob (torch.Tensor): the target
    """
    input_prob = input_prob.clamp(min=_EPS)
    target_prob = target_prob.clamp(min=_EPS)
    return F.kl_div(input_prob.log(), target_prob)


class SoftProbdist(nn.Module):
    def __init__(self, histogram_bins, temperature=1):
        super(SoftProbdist, self).__init__()
        self.register_buffer('bins', histogram_bins)
        self.scale = self.bins[1] - self.bins[0]
        self.temperature = temperature

    def forward(self, tensor):
        """
        Calculates a differentiable probability density function approximation of a tensor.
        Args:
            tensor (torch.Tensor): the input variable
        """
        return soft_probdist(tensor, self.bins, self.temperature)


def soft_probdist(tensor, bins=None, temperature=1):
    """
    Calculates a differentiable approximation of a probability distribution.
    Args:
         tensor (torch.Tensor): the input tensor.
         bins (torch.Tensor): the bins on which we calculate the probability distribution.
         temperature (float): the temperature of the approximation - how sharp is the gaussian.
           default value is 1, for best performance leave 1.
    """
    if bins is None:
        bins = _get_bins_symm(tensor.abs().max(), 1024, tensor.device)
    assert bins.dim() == 1, "The bins must be a sequential 1 dim tensor."
    scale = bins[1] - bins[0]
    tensor_, bins_ = tensor.view(1, -1), bins.unsqueeze(1)
    x = tensor_ - bins_
    x = torch.exp(-torch.abs((x * temperature / (0.5 * scale)) ** 2.))
    soft_hist = x.sum(dim=1).clamp(_EPS)
    probdist = soft_hist / soft_hist.sum()
    return ProbDist(probdist, bins)


def real_probdist(tensor, bins=None):
    if bins is None:
        bins = _get_bins_symm(tensor.abs().max(), 1024, tensor.device)
    assert bins.dim() == 1, "The bins must be a sequential 1 dim tensor."
    hist = torch.histc(tensor, bins.shape[0])
    probdist = hist / tensor.numel()
    return ProbDist(probdist, bins)


class ProbDist:
    def __init__(self, distribution, bins):
        self.distribution = distribution  # type: torch.Tensor
        self.bins = bins  # type: torch.Tensor

    def savefig(self, p_name, save_dir='./plots/'):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        filename = os.path.join(save_dir, '%s.distribution.png' % p_name)
        b, d = self.bins.detach().cpu().numpy(), self.distribution.detach().cpu().numpy()
        plt.plot(b, d)
        plt.title('%s Distribution.' % p_name)
        # extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(filename)
        plt.clf()


def _get_bins_symm(M, n_bins, device='cpu', centered=False):
    shift = 0.5 if not centered else 0
    scale = M / n_bins
    bins = (torch.arange(-n_bins//2, n_bins//2, dtype=torch.int32, device=device)+shift) * scale  # type: torch.Tensor
    return bins


def LPDL(tensor, n_bins, temperature=1.0, n_centers=255):
    """
    LPDL - Laplace Probability Distribution Lattice.
    Generates a target linear mixture of Laplace distributions out of tensor of values.
    Args:
        tensor (torch.Tensor): values
        n_bins (int): number of bins in the histogram
        temperature (float): controls the sharpness of the Laplace Distributions
        n_centers (int): number of laplace distributions
    """
    with torch.no_grad():
        M, d = tensor.abs().max(), tensor.device
        bins, centers = _get_bins_symm(M, n_bins, d), _get_bins_symm(M, n_centers, d, centered=True)
        bins, centers = bins.unsqueeze(0), centers.unsqueeze(1)
        b = centers[1] - centers[0]
        x = bins - centers
        x = torch.exp(-temperature * x.abs() / b) * torch.cos(np.pi * x / b).abs()
        x = x.sum(dim=0)
        x_dist = x / x.sum()
        probdist = ProbDist(x_dist, bins.view(*x_dist.shape))
        probdist.centers = centers
        return probdist


def MLPDL(tensor, n_bins, temperature=1.0, n_centers=255, waist=1.0):
    with torch.no_grad():
        M, d = tensor.abs().max(), tensor.device
        bins, centers = _get_bins_symm(M, n_bins, d), _get_bins_symm(M, n_centers, d, centered=True)
        BB, CC = torch.meshgrid(bins, centers)
        b = centers[1] - centers[0]
        x = BB - CC
        y = torch.exp(-torch.abs(CC / (waist*b)))
        x = torch.exp(-temperature * x.abs() / b) * torch.cos(np.pi * x / b).abs()
        x = x * y
        x = x.sum(dim=1)
        x_dist = x / x.sum()
        probdist = ProbDist(x_dist, bins.view(*x_dist.shape))
        probdist.centers = centers
        return probdist


_STR_TO_F = {
    'LPDL': LPDL,
    'MLPDL': MLPDL
}


def plot_distributions(model, include_bn=False, include_bias=False, save_dir='./plots/'):
    """
    Plots the weights distributions for the model.
    """
    weights = ((p_name, p) for p_name, p in model.named_parameters()
                    if (include_bn or 'bn' not in p_name) and (include_bias or 'bias' not in p_name))

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for i, (p_name, p) in enumerate(weights):
        probdist = real_probdist(p)
        probdist.savefig(p_name, save_dir)

