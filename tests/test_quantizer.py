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
import os
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict
import pytest

module_path = os.path.abspath(os.path.join('..'))
from distiller.quantization import Quantizer
from distiller.quantization.quantizer import QBits, _ParamToQuant
from distiller.quantization.quantizer import FP_BKP_PREFIX
from distiller import has_children


#############################
# Dummy modules
#############################

class DummyQuantLayer(nn.Module):
    def __init__(self, qbits):
        super(DummyQuantLayer, self).__init__()
        self.qbits = qbits

    def forward(self, *input):
        return input


class DummyWrapperLayer(nn.Module):
    def __init__(self, module, qbits):
        super(DummyWrapperLayer, self).__init__()
        self.qbits = qbits
        self.inner = module

    def forward(self, *input):
        return input


class DummyModel(nn.Sequential):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.add_module('conv1', nn.Conv2d(3, 16, 1))
        self.add_module('bn1', nn.BatchNorm2d(16))
        self.add_module('relu1', nn.ReLU())
        self.add_module('pool1', nn.MaxPool2d(2, 2))

        def gen_sub_module():
            sub_m = nn.Sequential()
            sub_m.add_module('conv1', nn.Conv2d(16, 32, 1))
            sub_m.add_module('bn1', nn.BatchNorm2d(32))
            sub_m.add_module('relu1', nn.ReLU())
            sub_m.add_module('pool1', nn.MaxPool2d(2, 2))
            sub_m.add_module('conv2', nn.Conv2d(32, 16, 1))
            sub_m.add_module('bn2', nn.BatchNorm2d(16))
            sub_m.add_module('relu2', nn.ReLU())
            sub_m.add_module('pool2', nn.MaxPool2d(2, 2))
            return sub_m

        self.add_module('sub1', gen_sub_module())
        self.add_module('sub2', gen_sub_module())

        self.add_module('fc', nn.Linear(16, 10))
        self.add_module('last_relu', nn.ReLU(10))

        # Use zeroed parameters to make it easier to validate our dummy quantization function
        for p in self.parameters():
            p.data = torch.zeros_like(p)


#############################
# Dummy Quantizer
#############################

def dummy_quantize_params(param, param_meta):
    return param + param_meta.num_bits


class DummyQuantizer(Quantizer):
    def __init__(self, model, optimizer=None,
                 bits_activations=None, bits_weights=None, bits_bias=None,
                 overrides=None,
                 train_with_fp_copy=False):
        super(DummyQuantizer, self).__init__(model, optimizer, bits_activations, bits_weights, bits_bias, overrides,
                                             train_with_fp_copy)

        self.replacement_factory[nn.Conv2d] = lambda module, name, qbits_map: DummyWrapperLayer(module, qbits_map[name])
        self.replacement_factory[nn.ReLU] = lambda module, name, qbits_map: DummyQuantLayer(qbits_map[name])
        self.param_quantization_fn = dummy_quantize_params


#############################
# Other utils
#############################

expected_type_replacements = {nn.Conv2d: DummyWrapperLayer, nn.ReLU: DummyQuantLayer}


def params_quantizable(module):
    return isinstance(module, (nn.Conv2d, nn.Linear))


def get_expected_qbits(model, qbits, expected_overrides):
    expected_qbits = {}
    post_prepare_changes = {}
    prefix = 'module.' if isinstance(model, torch.nn.DataParallel) else ''
    for orig_name, orig_module in model.named_modules():
        bits_a, bits_w, bits_b = expected_overrides.get(orig_name.replace(prefix, '', 1), qbits)
        if not params_quantizable(orig_module):
            bits_w = bits_b = None
        expected_qbits[orig_name] = QBits(bits_a, bits_w, bits_b)

        # We're testing replacement of module with container
        if isinstance(orig_module, nn.Conv2d):
            post_prepare_changes[orig_name] = QBits(bits_a, None, None)
            post_prepare_changes[orig_name + '.inner'] = expected_qbits[orig_name]

    return expected_qbits, post_prepare_changes


#############################
# Fixtures
#############################

@pytest.fixture(name='model')
def fixture_model():
    return DummyModel()


# TODO: Test optimizer modifications in 'test_model_prep'
@pytest.fixture(name='optimizer')
def fixture_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture(name='train_with_fp_copy', params=[False, True], ids=['fp_copy_off', 'fp_copy_on'])
def fixture_train_with_fp_copy(request):
    return request.param


@pytest.fixture(name='parallel', params=[False, True], ids=['parallel_off', 'parallel_on'])
def fixture_parallel(request):
    return request.param


#############################
# Tests
#############################

def test_no_quantization(model):
    m_orig = deepcopy(model)

    q = DummyQuantizer(model)
    assert all(qbits.acts is None and qbits.wts is None and qbits.bias is None for qbits in q.module_qbits_map.values())

    q.prepare_model()
    assert len(q.params_to_quantize) == 0
    assert all(type(q_module) == type(orig_module) for q_module, orig_module in zip(model.modules(), m_orig.modules()))

    q.quantize_params()
    assert all(torch.equal(q_param, orig_param) for q_param, orig_param in zip(model.parameters(), m_orig.parameters()))


def test_overrides_ordered_dict(model):
    with pytest.raises(TypeError, message='Expecting TypeError when overrides is not an OrderedDict'):
        DummyQuantizer(model, overrides={'bits':{'testing': '123'}})


@pytest.mark.parametrize(
    "qbits, bits_overrides, explicit_expected_overrides",
    [
        (QBits(8, 4, 32), OrderedDict(), {}),
        (QBits(8, 4, 32),
         OrderedDict([('conv1', {'acts': None, 'wts': None, 'bias': None}),
                      ('relu1', {'acts': None, 'wts': None, 'bias': None})]),
         {'conv1': QBits(None, None, None), 'relu1': QBits(None, None, None)}),
        (QBits(8, 8, 32),
         OrderedDict([('sub.*conv1', {'wts': 4}), ('sub.*conv2', {'acts': 4, 'wts': 4})]),
         {'sub1.conv1': QBits(8, 4, 32), 'sub1.conv2': QBits(4, 4, 32), 'sub2.conv1': QBits(8, 4, 32), 'sub2.conv2': QBits(4, 4, 32)}),
        (QBits(4, 4, 32),
         OrderedDict([('sub1\..*1', {'acts': 16, 'wts': 16}), ('sub1\..*', {'acts': 8, 'wts': 8})]),
         {'sub1.conv1': QBits(16, 16, 32), 'sub1.bn1': QBits(16, None, None),
          'sub1.relu1': QBits(16, None, None), 'sub1.pool1': QBits(16, None, None),
          'sub1.conv2': QBits(8, 8, 32), 'sub1.bn2': QBits(8, None, None),
          'sub1.relu2': QBits(8, None, None), 'sub1.pool2': QBits(8, None, None)}),
        (QBits(4, 4, 32),
         OrderedDict([('sub1\..*', {'acts': 8, 'wts': 8}), ('sub1\..*1', {'acts': 16, 'wts': 16})]),
         {'sub1.conv1': QBits(8, 8, 32), 'sub1.bn1': QBits(8, None, None),
          'sub1.relu1': QBits(8, None, None), 'sub1.pool1': QBits(8, None, None),
          'sub1.conv2': QBits(8, 8, 32), 'sub1.bn2': QBits(8, None, None),
          'sub1.relu2': QBits(8, None, None), 'sub1.pool2': QBits(8, None, None)}),
    ],
    ids=[
        'no_override',
        'simple_override',
        'pattern_override',
        'overlap_pattern_override_proper',  # "proper" ==> Specific pattern before broader pattern
        'overlap_pattern_override_wrong'    # "wrong" ==> Broad pattern before specific pattern, so specific pattern
                                            #             never actually matched
    ]
)
def test_model_prep(model, optimizer, qbits, bits_overrides, explicit_expected_overrides,
                    train_with_fp_copy, parallel):
    if parallel:
        model = torch.nn.DataParallel(model)
    m_orig = deepcopy(model)

    # Build expected QBits
    expected_qbits, post_prepare_changes = get_expected_qbits(model, qbits, explicit_expected_overrides)
    overrides = OrderedDict([('bits', deepcopy(bits_overrides))])

    # Initialize Quantizer
    q = DummyQuantizer(model, optimizer=optimizer,
                       bits_activations=qbits.acts, bits_weights=qbits.wts, bits_bias=qbits.bias,
                       overrides=overrides, train_with_fp_copy=train_with_fp_copy)

    # Check number of bits for quantization were registered correctly
    assert q.module_qbits_map == expected_qbits

    q.prepare_model()
    expected_qbits.update(post_prepare_changes)

    for ptq in q.params_to_quantize:
        assert params_quantizable(ptq.module)
        assert expected_qbits[ptq.module_name].wts is not None

        # Check parameter names are as expected
        assert ptq.q_attr_name in ['weight', 'bias']

        named_params = dict(ptq.module.named_parameters())
        if q.train_with_fp_copy:
            # Checking parameter replacement is as expected
            assert ptq.fp_attr_name == FP_BKP_PREFIX + ptq.q_attr_name
            assert ptq.fp_attr_name in named_params
            assert ptq.q_attr_name not in named_params
            # Making sure the following doesn't throw an exception,
            # so we know q_attr_name is still a buffer in the module
            getattr(ptq.module, ptq.q_attr_name)
        else:
            # Make sure we didn't screw anything up
            assert ptq.fp_attr_name == ptq.q_attr_name
            assert ptq.fp_attr_name in named_params

        # Check number of bits registered correctly
        expected_n_bits = expected_qbits[ptq.module_name].bias if ptq.q_attr_name == 'bias' else \
            expected_qbits[ptq.module_name].wts
        assert ptq.num_bits == expected_n_bits

    q_named_modules = dict(model.named_modules())
    orig_named_modules = dict(m_orig.named_modules())
    for orig_name, orig_module in orig_named_modules.items():
        # Check no module name from original model is missing
        assert orig_name in q_named_modules

        # Check module replacement is as expected
        q_module = q_named_modules[orig_name]
        expected_type = expected_type_replacements.get(type(orig_module))
        if expected_type is None or expected_qbits[orig_name] == QBits(None, None, None):
            assert type(orig_module) == type(q_module)
        else:
            assert type(q_module) == expected_type
            if expected_type == DummyWrapperLayer:
                assert expected_qbits[orig_name + '.inner'] == q_module.qbits
            else:
                assert expected_qbits[orig_name] == q_module.qbits


@pytest.mark.parametrize(
    "qbits, bits_overrides, explicit_expected_overrides",
    [
        (QBits(8, 8, 32),
         OrderedDict([('conv1', {'acts': None, 'wts': None}), ('relu1', {'acts': None, 'wts': None}),
                      ('sub.*conv1', {'acts': 8, 'wts': 4}), ('sub.*conv2', {'acts': 4, 'wts': 4})]),
         {'conv1': QBits(None, None, None), 'relu1': QBits(None, None, None),
          'sub1.conv1': QBits(8, 4, 32), 'sub1.conv2': QBits(4, 4, 32), 'sub2.conv1': QBits(8, 4, 32),
          'sub2.conv2': QBits(4, 4, 32)}),
    ]
)
def test_param_quantization(model, optimizer, qbits, bits_overrides, explicit_expected_overrides,
                            train_with_fp_copy):
    # Build expected QBits
    expected_qbits, post_prepare_changes = get_expected_qbits(model, qbits, explicit_expected_overrides)
    overrides = OrderedDict([('bits', deepcopy(bits_overrides))])

    q = DummyQuantizer(model, optimizer=optimizer,
                       bits_activations=qbits.acts, bits_weights=qbits.wts, bits_bias=qbits.bias,
                       overrides=overrides, train_with_fp_copy=train_with_fp_copy)
    q.prepare_model()
    expected_qbits.update(post_prepare_changes)

    q_model_pre_quant = deepcopy(model)
    q.quantize_params()
    for (name, pre_quant_module), post_quant_module in zip(q_model_pre_quant.named_modules(), model.modules()):
        # Skip containers
        # if len(list(pre_quant_module.modules())) > 1:
        if has_children(pre_quant_module):
            continue

        num_qbits = expected_qbits[name].wts

        for param_name, pre_quant_param in pre_quant_module.named_parameters():
            quantizable = num_qbits is not None
            if param_name.endswith('bias'):
                num_bits = expected_qbits[name].bias
            else:
                num_bits = num_qbits

            if quantizable and train_with_fp_copy:
                # "param_name" and "pre_quant_param" refer to the float copy

                # Check the float copy didn't change
                post_quant_fp_copy = getattr(post_quant_module, param_name)
                assert torch.equal(pre_quant_param, post_quant_fp_copy)

                quant_param = getattr(post_quant_module, param_name.replace(FP_BKP_PREFIX, ''))

                # Check weights quantization properly recorded for autograd
                gfn = quant_param.grad_fn
                assert gfn is not None
                assert str(type(gfn).__name__) == 'AddBackward0'
                gfn = gfn.next_functions[0][0]
                assert str(type(gfn).__name__) == 'AccumulateGrad'
                assert id(gfn.variable) == id(post_quant_fp_copy)
            else:
                quant_param = getattr(post_quant_module, param_name)

            expected = dummy_quantize_params(pre_quant_param,
                                             _ParamToQuant(None, None, None, None, num_bits)) if quantizable else pre_quant_param
            assert torch.equal(quant_param, expected)
