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
import torch
import os
import sys
import pytest
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
from models import ALL_MODEL_NAMES, create_model
from apputils import *
from distiller import normalize_module_name, denormalize_module_name

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def get_input(dataset):
    if dataset == 'imagenet':
        return torch.randn((1, 3, 224, 224), requires_grad=False)
    elif dataset == 'cifar10':
        return torch.randn((1, 3, 32, 32))
    return None


def create_graph(dataset, arch):
    dummy_input = get_input(dataset)
    assert dummy_input is not None, "Unsupported dataset ({}) - aborting draw operation".format(dataset)

    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None
    return SummaryGraph(model, dummy_input)


def test_graph():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None


def test_connectivity():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    op_names = [op['name'] for op in g.ops.values()]
    assert 73 == len(op_names)

    edges = g.edges
    assert edges[0].src == '0' and edges[0].dst == 'conv1'

    # Test two sequential calls to predecessors (this was a bug once)
    preds = g.predecessors(g.find_op('bn1'), 1)
    preds = g.predecessors(g.find_op('bn1'), 1)
    assert preds == ['108', '2', '3', '4', '5']
    # Test successors
    succs = g.successors(g.find_op('bn1'), 2)
    assert succs == ['relu']

    op = g.find_op('layer1.0')
    assert op is not None
    preds = g.predecessors(op, 2)
    assert preds == ['layer1.0.bn2', 'relu']

    op = g.find_op('layer1.0.relu2')
    assert op is not None
    succs = g.successors(op, 4)
    assert succs == ['layer1.1.bn1', 'layer1.1.relu2']

    preds = g.predecessors(g.find_op('bn1'), 10)
    assert preds == []
    preds = g.predecessors(g.find_op('bn1'), 3)
    assert preds == ['0', '1']


def test_layer_search():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    op = g.find_op('layer1.0.conv1')
    assert op is not None

    succs = g.successors_f('layer1.0.conv1', 'Conv', [], logging)
    assert ['layer1.0.conv2'] == succs

    succs = g.successors_f('relu', 'Conv', [], logging)
    assert succs == ['layer1.0.conv1', 'layer1.1.conv1', 'layer1.2.conv1', 'layer2.0.conv1', 'layer2.0.downsample.0']

    succs = g.successors_f('relu', 'Gemm', [], logging)
    assert succs == ['fc']

    succs = g.successors_f('layer3.2', 'Conv', [], logging)
    assert succs == []
    #logging.debug(succs)

    preds = g.predecessors_f('conv1', 'Conv', [], logging)
    assert preds == []

    preds = g.predecessors_f('layer1.0.conv2', 'Conv', [], logging)
    assert preds == ['layer1.0.conv1']

    preds = g.predecessors_f('layer1.0.conv1', 'Conv', [], logging)
    assert preds == ['conv1']

    preds = g.predecessors_f('layer1.1.conv1', 'Conv', [], logging)
    assert preds == ['layer1.0.conv2', 'conv1']


def test_vgg():
    g = create_graph('imagenet', 'vgg19')
    assert g is not None
    succs = g.successors_f('features.32', 'Conv')
    logging.debug(succs)
    succs = g.successors_f('features.34', 'Conv')


def test_simplenet():
    g = create_graph('cifar10', 'simplenet_cifar')
    assert g is not None
    preds = g.predecessors_f(normalize_module_name('module.conv1'), 'Conv')
    logging.debug("[simplenet_cifar]: preds of module.conv1 = {}".format(preds))
    assert len(preds) == 0

    preds = g.predecessors_f(normalize_module_name('module.conv2'), 'Conv')
    logging.debug("[simplenet_cifar]: preds of module.conv2 = {}".format(preds))
    assert len(preds) == 1


def name_test(dataset, arch):
    model = create_model(False, dataset, arch, parallel=False)
    modelp = create_model(False, dataset, arch, parallel=True)
    assert model is not None and modelp is not None

    mod_names   = [mod_name for mod_name, _ in model.named_modules()]
    mod_names_p = [mod_name for mod_name, _ in modelp.named_modules()]
    assert mod_names is not None and mod_names_p is not None
    assert len(mod_names)+1 == len(mod_names_p)

    for i in range(len(mod_names)-1):
        assert mod_names[i+1] == normalize_module_name(mod_names_p[i+2])
        logging.debug("{} {} {}".format(mod_names_p[i+2], mod_names[i+1], normalize_module_name(mod_names_p[i+2])))
        assert mod_names_p[i+2] == denormalize_module_name(modelp, mod_names[i+1])


def test_normalize_module_name():
    assert "features.0" == normalize_module_name("features.module.0")
    assert "features.0" == normalize_module_name("module.features.0")
    assert "features" == normalize_module_name("features.module")
    name_test('imagenet', 'vgg19')
    name_test('cifar10', 'resnet20_cifar')
    name_test('imagenet', 'alexnet')


def test_onnx_name_2_pytorch_name():
    assert "layer3.0.relu1" == onnx_name_2_pytorch_name("ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu].1", 'Relu')
    assert "features.34" == onnx_name_2_pytorch_name('VGG/[features]/Sequential/Conv2d[34]', 'Conv')
    assert "Relu3" == onnx_name_2_pytorch_name('NameWithNoModule.3', 'Relu')
    #assert "features.module.34" == onnx_name_2_pytorch_name('VGG/DataParallel[features]/Sequential/Conv2d[34]', 'Conv')


def test_connectivity_summary():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    summary = connectivity_summary(g)
    assert len(summary) == 73

    verbose_summary = connectivity_summary_verbose(g)
    assert len(verbose_summary  ) == 73


if __name__ == '__main__':
    test_connectivity_summary()
