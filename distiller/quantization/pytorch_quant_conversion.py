import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from copy import deepcopy
from collections import OrderedDict

import distiller
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


def distiller_quantized_tensor_to_pytorch(t: torch.Tensor, scale, zp, num_bits, distiller_mode, dest_dtype,
                                          per_ch=False, ch_dim=0):
    assert (t == t.int()).all(), 'Tensor does not appear to be quantized'
    converted_scale, converted_zp = distiller_qparams_to_pytorch(scale, zp, num_bits, distiller_mode, dest_dtype)
    zp_diff = -converted_zp.view(zp.shape) - zp

    if dest_dtype == torch.quint8:
        temp_dtype = torch.uint8
    elif dest_dtype == torch.qint8:
        temp_dtype = torch.int8
    else:  # dest_dtype == torch.qint32:
        temp_dtype = torch.int32
    t = (t - zp_diff).to(temp_dtype)
    if per_ch:
        return torch._make_per_channel_quantized_tensor(t, converted_scale, converted_zp, ch_dim)
    return torch._make_per_tensor_quantized_tensor(t, converted_scale, converted_zp)


def _ptq_convert_pass_replace_range_linear_wrappers(module):
    # Hacky deferred import
    from distiller.quantization import RangeLinearQuantWrapper

    reassign = OrderedDict()
    for n, m in module.named_children():
        new_m = m
        if isinstance(m, distiller.quantization.RangeLinearQuantWrapper):
            new_m = m.to_pytorch_quant()

            requires_quantized_inputs = not (isinstance(new_m, nn.Sequential) and
                                             isinstance(new_m[0], ConditionalDeQuantizeWrapper))

            if requires_quantized_inputs:
                d = OrderedDict()
                for idx, qmd in m.inputs_quant_metadata_fallback.items():
                    qset = m.inputs_quant_settings_overrides.get(idx, m.output_quant_settings)
                    scale, zp = distiller_qparams_to_pytorch(
                        qmd.scale, qmd.zero_point, qset.num_bits, qset.quant_mode, torch.quint8
                    )
                    d[idx] = (scale, zp, torch.quint8)
                new_m = ConditionalQuantizeWrapper(new_m, d)
        elif distiller.has_children(m):
            new_m = _ptq_convert_pass_replace_range_linear_wrappers(m)
        elif not isinstance(m, nn.Identity):
            # Module not quantized in Distiller, possibly need to de-quant input
            new_m = ConditionalDeQuantizeWrapper(m)
        reassign[n] = new_m

    for n, new_m in reassign.items():
        module._modules[n] = new_m

    return module


def _ptq_convert_pass_remove_redundant_quant_dequant(model, dummy_input):
    def quantize_wrapper_check_hook(module, inputs):
        if not isinstance(module, ConditionalQuantize):
            return
        q_inputs = []
        for idx, t in enumerate(inputs):
            if not isinstance(t, torch.Tensor):
                continue
            if t.is_quantized:
                q_inputs.append(idx)
        module.already_quantized = q_inputs

    def dequant_wrapper_check_hook(module, input):
        if not isinstance(module, ConditionalDeQuantize):
            return
        module.any_quantized = False

        def check_recursively(x):
            if isinstance(x, torch.Tensor) and x.is_quantized:
                module.any_quantized = True
            elif isinstance(x, (tuple, list)):
                for item in x:
                    check_recursively(item)

        check_recursively(input)

    handles = []
    for m in model.modules():
        if isinstance(m, ConditionalQuantize):
            handles.append(m.register_forward_pre_hook(quantize_wrapper_check_hook))
        elif isinstance(m, ConditionalDeQuantize):
            handles.append(m.register_forward_pre_hook(dequant_wrapper_check_hook))
    model(dummy_input)
    for h in handles:
        h.remove()

    def cleanup(module):
        reassign = OrderedDict()
        for n, m in module.named_children():
            new_m = m
            if isinstance(m, ConditionalQuantizeWrapper):
                for idx in m.quant.already_quantized:
                    m.quant.quantizers.pop(str(idx))
                if len(m.quant.quantizers) == 0:
                    new_m = m.wrapped
            elif isinstance(m, ConditionalDeQuantizeWrapper):
                if not m.dequant.any_quantized:
                    new_m = m.wrapped
            elif distiller.has_children(m):
                cleanup(m)
            reassign[n] = new_m
        for n, new_m in reassign.items():
            module._modules[n] = new_m

        return module

    model = cleanup(model)
    return model


def convert_distiller_ptq_model_to_pytorch(model, dequant_output=True, dummy_input=None, inplace=False):
    if not inplace:
        model = deepcopy(model)

    model = _ptq_convert_pass_replace_range_linear_wrappers(model)
    if dummy_input is not None:
        model = _ptq_convert_pass_remove_redundant_quant_dequant(model, dummy_input)

    if dequant_output:
        model = nn.Sequential(OrderedDict([('model', model), ('dequant', nnq.DeQuantize())]))

    return model


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


class ConditionalDeQuantize(nn.Module):
    def __init__(self):
        super(ConditionalDeQuantize, self).__init__()

    def forward(self, *inputs):
        def dequant_recursively(x):
            if isinstance(x, torch.Tensor):
                return x.dequantize() if x.is_quantized else x
            if isinstance(x, (tuple, list)):
                return type(x)(dequant_recursively(item) for item in x)
            return x
        outputs = dequant_recursively(inputs)
        return outputs


class ConditionalDeQuantizeWrapper(nn.Module):
    def __init__(self, wrapped_module):
        super(ConditionalDeQuantizeWrapper, self).__init__()
        self.dequant = ConditionalDeQuantize()
        self.wrapped = wrapped_module

    def forward(self, *inputs):
        out = self.dequant(*inputs)
        out = self.wrapped(*out)
        return out


class ConditionalQuantize(nn.Module):
    def __init__(self, inputs_to_qparams_map):
        super(ConditionalQuantize, self).__init__()
        self.quantizers = nn.ModuleDict()
        for idx, qparams in inputs_to_qparams_map.items():
            self.quantizers[str(idx)] = nnq.Quantize(*qparams)

    def forward(self, *inputs):
        q_inputs = []
        for idx, item in enumerate(inputs):
            idx_str = str(idx)
            if idx_str in self.quantizers:
                assert isinstance(item, torch.Tensor), 'Trying to quantize a non-Tensor object'
                if not item.is_quantized:
                    item = self.quantizers[idx_str](item)
            q_inputs.append(item)
        # return q_inputs[0] if len(q_inputs) == 1 else tuple(q_inputs)
        return tuple(q_inputs)


class ConditionalQuantizeWrapper(nn.Module):
    def __init__(self, wrapped_module, inputs_to_qparams_map):
        super(ConditionalQuantizeWrapper, self).__init__()
        self.quant = ConditionalQuantize(inputs_to_qparams_map)
        self.wrapped = wrapped_module

    def forward(self, *inputs):
        out = self.quant(*inputs)
        out = self.wrapped(*out)
        return out
