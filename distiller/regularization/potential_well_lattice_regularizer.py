"""
PotentialWellLatticeRegularizer:
    A Lattice of potential wells as regularization policy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .regularizer import _Regularizer
from copy import deepcopy

EPS = 1e-6

__all__ = ['Lattice', 'PotentialWellLatticeLoss', 'PotentialWellLatticeRegularizer']


def _is_scalar(t):
    if np.isscalar(t):
        return True
    if isinstance(t, torch.Tensor) and t.dim() == 0:
        return True
    return False


def _broadcast_centers(tensor: torch.Tensor, centers: torch.Tensor, dim=None):
    # TODO - implement per-channel centers+amplitudes.
    dim = dim or -1
    tensor = tensor.unsqueeze(dim)
    broadcast_shape = [1] * tensor.dim()
    broadcast_shape[dim] = centers.size(0)
    centers = centers.view(*broadcast_shape)
    return tensor, centers


def _get_centers(range_bounds, n_centers):
    m, M = range_bounds
    assert m < M, 'The maximum value should be larger than the minimum value.'
    return (torch.arange(n_centers).float() + 0.5) / (M - m) + m


def _get_amplitude_logits(tensor, n_centers, range_bounds=(0, 0)):
    m, M = range_bounds
    histc = torch.histc(tensor, bins=n_centers, min=m, max=M)
    probdist = histc / histc.sum()
    probdist.clamp_min_(EPS)
    logits = probdist.log()
    return logits


class Lattice:
    """
    Represents a lattice environment, with uniform intervals between the centers.
    Each point in the lattice also has a different potential energy in its proximity
    represented by amplitude_logits.
    Args:
        n_lattice (int or None): the amount of centers in the lattice.
        value_bounds (tuple[scalar, scalar] or None): the bounds of the potential well.
        init_tensor (torch.Tensor or None): the initial tensor to calculate the centers+amplitudes of the
          potential well.
        dim (int or None): the dimension along which we compute the potential energies.
    Attributes:
        centers (torch.Tensor): the centers of the unit cells in the lattice.
        amplitude_logits (torch.Tensor): the "potential energy" around the centers.
    Note:
        centers and amplitude_logits must be of the same shape!

    TODO - implement per-channel centers+amplitudes.
    """
    DEFAULT_VALUE_BOUNDS = (-1., 1.)

    def __init__(self, n_lattice=None, value_bounds=None, init_tensor=None,
                 centers=None, amplitude_logits=None, dim=None):
        self.n_lattice = n_lattice
        self.dim = dim
        self._value_bounds = value_bounds
        self._init_tensor = init_tensor.clone.detach()
        self._check_args()

        if not centers and not amplitude_logits:
            if not n_lattice:
                raise ValueError('Please provide either a lattice configuration using n_lattice(int>0)'
                                 '[[, value_bounds][, init_tensor]] '
                                 'or the lattice vectors themselves using centers and amplitude_logits.')
            self._init_lattice()
        else:
            if not isinstance(centers, torch.Tensor) or not isinstance(amplitude_logits, torch.Tensor):
                raise TypeError('Lattice must be composed of two torch.Tensors: centers and amplitude_logits.')
            if not centers.shape == amplitude_logits.shape:
                raise ValueError('Size mismatch - lattice parameters must have the same 1D shape. '
                                 '(%s != %s)' % (centers.shape, amplitude_logits.shape))
            if not centers.dim() != 1:
                raise ValueError('Lattice dimension should be 1.')
            assert centers.device == amplitude_logits.device, \
                'centers & amplitude_logits must be on the same device.'

            self.centers = centers
            self.amplitude_logits = amplitude_logits
            self.n_lattice = self.centers.numel()

    def _check_args(self):
        assert self.n_lattice is None or isinstance(self.n_lattice, int) and self.n_lattice > 0, \
            'n_lattice must be a positive integer.'

        assert self._value_bounds is None or \
            (len(self._value_bounds) == 2 and all(_is_scalar(i) for i in self._value_bounds)), \
            'value_bounds must be a tuple of two scalars (int or float or tensor of dim 0).'

        assert self._init_tensor is None or \
            isinstance(self._init_tensor, torch.Tensor), 'init_tensor must be a Tensor.'

    def _init_lattice(self):
        if not self.n_lattice:
            raise ValueError('n_lattice must be initialized if centers & amplitude_logits weren\'t provided.')
        if not self._value_bounds:
            if self._init_tensor:
                self._value_bounds = self._init_tensor.min(dim=self.dim), self._init_tensor.max(dim=self.dim)
            else:
                self._value_bounds = PotentialWellLatticeLoss.DEFAULT_VALUE_BOUNDS
        if self._init_tensor is None:
            self._init_tensor = torch.linspace(*self._value_bounds, steps=4 * self.n_lattice)
        self.centers = _get_centers(self._value_bounds, self.n_lattice)
        self.amplitude_logits = nn.Parameter(_get_amplitude_logits(self._init_tensor,
                                                                   self.n_lattice, self._value_bounds))

    def amplitudes(self, dim=None):
        return F.softmax(self.amplitude_logits, dim=dim)

    @property
    def unit_cell_size(self):
        return self.centers[1] - self.centers[0]

    def to(self, *args, **kwargs):
        self.centers = self.centers.to(*args, **kwargs)
        self.amplitude_logits = self.amplitude_logits.to(*args, **kwargs)

    @property
    def device(self):
        return self.centers.device

    @device.setter
    def device(self, v):
        self.centers.device = v
        self.amplitude_logits.device = v


class PotentialWellLatticeLoss(nn.Module):
    KERNELS = ['mse', 'gaussian', 'laplace', 'sharp_periodic_gaussian', 'sharp_periodic_laplace']
    _KERNEL_METHOD_NAMES = ['_%s_kernel' % name for name in KERNELS]

    def __init__(self, potential_kernel, dim=None, **kernel_kwargs):
        """
        Represents the potential energy in a lattice environment, similar to the Nearly Free Electron model in
        solid-state physics. The energy is modeled with a lattice and
        a potential kernel w.r.t. distance around the centers.

        Args:
            dim (int): the dimension on which to apply this loss.
            distance_kernel (callable or str): A function that gives a positive penalty over larger distances.
              if str - pick from some predefined functions.
            distance_kernel_kwargs (dict): the keyword args for the kernel.
        """
        super().__init__()
        self.kernel_kwargs = kernel_kwargs
        self.dim = dim

        self._KERNEL_METHODS = {name: getattr(self, method_name)
                                for name, method_name in
                                zip(self.KERNELS, self._KERNEL_METHOD_NAMES)}
        if isinstance(potential_kernel, str):
            potential_kernel = self._KERNEL_METHODS[potential_kernel]
        if not callable(potential_kernel):
            raise TypeError('potential_kernel must either be a callable(x, target, **kwargs) or a string '
                            'representing a method. \n'
                            'Available kernels: %s' % self.KERNELS)
        self.potential_kernel = potential_kernel

    def forward(self, x, lattice: Lattice):
        amplitudes = lattice.amplitudes(self.dim)
        xx, cc = _broadcast_centers(x, lattice.centers)
        aa = amplitudes.view(*cc.shape)
        distance_penalties = self.potential_kernel(xx, cc)
        tuned_distance_penalties = aa * distance_penalties  # type: torch.Tensor
        return tuned_distance_penalties.mean()

    def _mse_kernel(self, x, target):
        """
        Kernel for squared distance.
        kwargs:
            scale (float): the scale for the distance.
        Note:
            The scale keyword argument is *redundant* because this scale can be controlled using
              the 'reg_regims' argument for the regularizer class.
        """
        scale = self.kernel_kwargs.get('scale', 1)
        return F.mse_loss(x, target, reduce=False) / scale ** 2

    def _gaussian_kernel(self, x, target):
        """
        Kernel for gaussian function.
        kwargs:
            sigma (float): std for gaussian function.
        """
        sigma = self.kernel_kwargs.get('sigma', 1)
        return (1 - torch.exp(-F.mse_loss(x, target, reduce=False) / (2 * sigma ** 2))) / (sigma * (2 * np.pi) ** 0.5)

    def _laplace_kernel(self, x, target):
        """
        Laplace kernel.
        kwargs:
            b (float): scale parameter for laplace function.
        """
        b = self.kernel_kwargs.get('b', 1)
        return (1 - torch.exp(-torch.abs(x - target) / b)) / (2 * b)

    def _sharp_periodic_gaussian_kernel(self, x, target):
        """
        Sharp-periodic gaussian kernel: 1 - |cos((distance/sigma) * pi)|*gaussian(distance;sigma)
        kwargs:
            sigma (float): std for gaussian function.
            k (float): the inverse lattice unit cell size.
        Note:
            the optimal parameters for this kernel is:
              sigma << lattice.unit_cell_size (sigma->0 yields dirac's delta distribution!)
              k = pi / lattice.unit_cell_size
        """
        sigma = self.kernel_kwargs.get('sigma', 1)
        k = self.kernel_kwargs.get('k', 1)
        err = x - target
        teeth = torch.abs(torch.cos(err * k))
        gaussian = torch.exp(-F.mse_loss(x, target, reduce=False) / (2 * sigma ** 2))
        return (1 - gaussian * teeth) / (sigma * (2 * np.pi) ** 0.5)

    def _sharp_periodic_laplace_kernel(self, x, target):
        """
        Sharp-periodic laplace kernel: 1-|cos((distance/sigma) * pi)|*laplace(distance;b)
        kwargs:
            b (float): scale parameter for laplace function.
            k (float): the inverse lattice unit cell size.
        Note:
            the optimal parameters for this kernel is:
              b << lattice.unit_cell_size (b->0 yields dirac's delta distribution!)
              k = pi / lattice.unit_cell_size
        """
        b = self.kernel_kwargs.get('b', 1)
        k = self.kernel_kwargs.get('k', 1)
        err = x - target
        teeth = torch.abs(torch.cos(err * k))
        laplace = torch.exp(-torch.abs(err) / b)
        return (1 - laplace * teeth) / (2 * b)

    def get_help_kernel(self, kernel_name=None):
        if kernel_name:
            method = self._KERNEL_METHODS.get(kernel_name, None)
        else:
            method = self.potential_kernel
        if method is None:
            raise NameError('No builtin kernel named \'%s\'. \n '
                            'Builtin kernels: %s' % (kernel_name, self.KERNELS))
        return method.__doc__


class PotentialWellLatticeRegularizer(_Regularizer):
    def __init__(self, name, model, reg_regims, lattices_config):
        """
        Regularizer of potential well lattice.
        Args:
            name (str): the name of the model.
            model (nn.Module): the model.
            reg_regims (dict[str, tuple[float, dict]]): regularizer regimes for all parameters included.
            lattices_config (dict[str, dict]): configuration for lattices for each of the parameters.
              see Lattice.__init__ for details on accepted arguments.
        Note:
            1. The use of centers+amplitude_logits arguments is not allowed using this API, however using
              the other arguments you can achieve the same result.
            2. Using lattices_config you can specify the key 'all' to allow global default configuration, e.g.:
              lattices_config = {
                  'all': {'n_lattice': 4, 'value_bounds': [-1,1]},
                  'module.fc.weight' : {'n_lattice': 8, 'value_bounds': None}  # overrides the global settings.
              }

              same can be done with reg_regims!
        """
        super().__init__(name, model, reg_regims, None)
        self.lattices_config = lattices_config
        default_lattice_config = lattices_config.get('all', {})
        self.lattices = {}  # type: dict[str, Lattice]
        self.losses_modules = {}    # type: dict[str, PotentialWellLatticeLoss]
        losses_config = {name: conf for name, (loss_weight, conf) in self.reg_regims.items()}
        default_loss_config = losses_config.get('all', {})
        for param_name, param in model.named_parameters():
            # get local configuration
            _lattice_config, _loss_config = lattices_config.get(param_name, {}), losses_config.get(param_name, {})
            lattice_config, loss_config = deepcopy(default_lattice_config), deepcopy(default_loss_config)
            # override default configuration with local configuration
            lattice_config.update(_lattice_config)
            loss_config.update(_loss_config)
            self.lattices[param_name] = Lattice(init_tensor=param, **lattice_config)
            self.losses_modules[param_name] = PotentialWellLatticeLoss(**loss_config)

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        loss_fn = self.losses_modules[param_name]
        lattice = self.lattices[param_name]
        loss = loss_fn(param, lattice)
        regularizer_loss += loss
        return regularizer_loss

    def threshold(self, param, param_name, zeros_mask_dict):
        raise NotImplementedError
