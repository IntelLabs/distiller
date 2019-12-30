import torch
import torch.nn as nn
import torch.nn.quantized as nnq

from .q_utils import LinearQuantMode


def distiller_qparams_to_pytorch(scale, zp, num_bits, distiller_mode, dest_dtype):
    assert dest_dtype in (torch.qint8, torch.quint8)

    scale = scale.cpu().squeeze()
    zp = zp.cpu().squeeze().long()

    # Distiller scale is the reciprocal of PyTorch scale
    scale_torch = 1. / scale

    n_bins_half = 2 ** (num_bits - 1)

    if distiller_mode == LinearQuantMode.SYMMETRIC:
        # In Distiller symmetric is always signed with zero-point = 0, but in PyTorch it can be
        # unsigned in which case we offset the zero-point to the middle of the quantized range
        zp_torch = zp if dest_dtype == torch.qint8 else torch.full_like(zp, n_bins_half)
    else:
        distiller_signed = distiller_mode == LinearQuantMode.ASYMMETRIC_SIGNED
        pytorch_signed = dest_dtype == torch.qint8
        if distiller_signed and not pytorch_signed:
            zp = zp - n_bins_half
        elif not distiller_signed and pytorch_signed:
            zp = zp + n_bins_half
        # Distiller subtracts the zero-point when quantizing, PyTorch adds it.
        # So we negate the zero-point calculated in Distiller
        zp_torch = -zp
    return scale_torch, zp_torch


def distiller_ptq_tensor_to_pytorch(t, scale, zp, dtype, per_ch=False, ch_dim=0):
    if dtype == torch.quint8:
        dtype = torch.uint8
    elif dtype == torch.qint8:
        dtype = torch.int8
    elif dtype == torch.qint32:
        dtype = torch.int32
    if per_ch:
        return torch._make_per_channel_quantized_tensor(t.to(dtype), scale, zp, ch_dim)
    return torch._make_per_tensor_quantized_tensor(t.to(dtype), scale, zp)


class QFunctionalWrapper(nn.Module):
    def __init__(self):
        super(QFunctionalWrapper, self).__init__()
        self.qfunc = nnq.QFunctional()


class QFunctionalAdd(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAdd, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add(x, y)


class QFunctionalAddScalar(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAddScalar, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add_scalar(x, y)


class QFunctionalMul(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalMul, self).__init__()

    def forward(self, x, y):
        return self.qfunc.mul(x, y)


class QFunctionalMulScalar(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalMulScalar, self).__init__()

    def forward(self, x, y):
        return self.qfunc.mul_scalar(x, y)


class QFunctionalCat(QFunctionalWrapper):
    def __init__(self, dim=0):
        super(QFunctionalCat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return self.qfunc.cat(x, self.dim)


class QFunctionalAddRelu(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAddRelu, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add_relu(x, y)


class ConditionalDeQuantizeWrapper(nn.Module):
    def __init__(self, wrapped_module):
        super(ConditionalDeQuantizeWrapper, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, *inputs):
        def dequant_recursively(x):
            if isinstance(x, torch.Tensor):
                return x.dequantize() if x.is_quantized else x
            if isinstance(x, (tuple, list)):
                return type(x)(dequant_recursively() for item in x)
            return x
        inputs = dequant_recursively(inputs)
        return self.wrapped_module(*inputs)


class ConditionalQuantizeWrapper(nn.Module):
    def __init__(self, wrapped_module, inputs_to_qparams_map):
        super(ConditionalQuantizeWrapper, self).__init__()
        self.inputs_to_qparams_map = inputs_to_qparams_map
        self.wrapped_module = wrapped_module

    def forward(self, *inputs):
        q_inputs = []
        for idx, item in enumerate(inputs):
            qparams = self.inputs_to_qparams_map.get(idx, None)
            if qparams:
                assert isinstance(item, torch.Tensor), 'Trying to quantize a non-Tensor object'
                if not item.is_quantized:
                    item = torch.quantize_per_tensor(item, *qparams)
            q_inputs.append(item)
        return self.wrapped_module(*q_inputs)
