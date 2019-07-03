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
import torch.nn as nn
import pytest
import distiller
from distiller.models import ALL_MODEL_NAMES, create_model
from distiller.apputils import *
from distiller import normalize_module_name, denormalize_module_name, \
    SummaryGraph, onnx_name_2_pytorch_name
from distiller.model_summaries import connectivity_summary, connectivity_summary_verbose
from distiller.summary_graph import AdjacentsEntry, OpSimpleMetadata

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def create_graph(dataset, arch, parallel=False):
    dummy_input = distiller.get_dummy_input(dataset)
    model = create_model(False, dataset, arch, parallel)
    assert model is not None
    return SummaryGraph(model, dummy_input)


def test_graph():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None


@pytest.fixture(params=[False, True], ids=['sequential', 'parallel'])
def parallel(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['denorm_name', 'norm_name'])
def denorm_names(request):
    return request.param


def prefix_strs(str_list, prefix):
    return [prefix + s for s in str_list]


def test_connectivity(parallel, denorm_names):
    g = create_graph('cifar10', 'resnet20_cifar', parallel)
    assert g is not None

    prefix = 'module.' if parallel and denorm_names else ''

    op_names = [op['name'] for op in g.ops.values()]
    assert len(op_names) == 80

    edges = g.edges
    assert edges[0].src == '0' and edges[0].dst == 'conv1'

    # Test two sequential calls to predecessors (this was a bug once)
    preds = g.predecessors(g.find_op('bn1'), 1, denorm_names=denorm_names)
    preds = g.predecessors(g.find_op('bn1'), 1, denorm_names=denorm_names)
    assert preds == ['129', '2', '3', '4', '5']
    # Test successors
    succs = g.successors(g.find_op('bn1'), 2, denorm_names=denorm_names)
    assert succs == prefix_strs(['relu'], prefix)

    op = g.find_op('layer1.0.relu2')
    assert op is not None
    succs = g.successors(op, 4, denorm_names=denorm_names)
    assert succs == prefix_strs(['layer1.1.bn1', 'layer1.1.relu2'], prefix)

    preds = g.predecessors(g.find_op('bn1'), 10, denorm_names=denorm_names)
    assert preds == []
    preds = g.predecessors(g.find_op('bn1'), 3, denorm_names=denorm_names)
    assert preds == ['0', '1']


def test_layer_search(parallel, denorm_names):
    g = create_graph('cifar10', 'resnet20_cifar', parallel)
    assert g is not None

    prefix = 'module.' if parallel and denorm_names else ''

    op = g.find_op('layer1.0.conv1')
    assert op is not None

    succs = g.successors_f('layer1.0.conv1', 'Conv', [], logging, denorm_names=denorm_names)
    assert succs == prefix_strs(['layer1.0.conv2'], prefix)

    succs = g.successors_f('relu', 'Conv', [], logging, denorm_names=denorm_names)
    assert succs == prefix_strs(['layer1.0.conv1', 'layer1.1.conv1', 'layer1.2.conv1', 'layer2.0.conv1',
                                 'layer2.0.downsample.0'], prefix)

    succs = g.successors_f('relu', 'Gemm', [], logging, denorm_names=denorm_names)
    assert succs == prefix_strs(['fc'], prefix)

    succs = g.successors_f('layer3.2', 'Conv', [], logging, denorm_names=denorm_names)
    assert succs == []

    preds = g.predecessors_f('conv1', 'Conv', [], logging, denorm_names=denorm_names)
    assert preds == []

    preds = g.predecessors_f('layer1.0.conv2', 'Conv', [], logging, denorm_names=denorm_names)
    assert preds == prefix_strs(['layer1.0.conv1'], prefix)

    preds = g.predecessors_f('layer1.0.conv1', 'Conv', [], logging, denorm_names=denorm_names)
    assert preds == prefix_strs(['conv1'], prefix)

    preds = g.predecessors_f('layer1.1.conv1', 'Conv', [], logging, denorm_names=denorm_names)
    assert preds == prefix_strs(['layer1.0.conv2', 'conv1'], prefix)


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


def test_normalize_module_name():
    def name_test(dataset, arch):
        model = create_model(False, dataset, arch, parallel=False)
        modelp = create_model(False, dataset, arch, parallel=True)
        assert model is not None and modelp is not None

        mod_names = [mod_name for mod_name, _ in model.named_modules()]
        mod_names_p = [mod_name for mod_name, _ in modelp.named_modules()]
        assert mod_names is not None and mod_names_p is not None
        assert len(mod_names) + 1 == len(mod_names_p)

        for i in range(len(mod_names) - 1):
            assert mod_names[i + 1] == normalize_module_name(mod_names_p[i + 2])
            logging.debug(
                "{} {} {}".format(mod_names_p[i + 2], mod_names[i + 1], normalize_module_name(mod_names_p[i + 2])))
            assert mod_names_p[i + 2] == denormalize_module_name(modelp, mod_names[i + 1])

    assert normalize_module_name("features.module.0") == "features.0"
    assert normalize_module_name("module.features.0") == "features.0"
    assert normalize_module_name("features.module") == "features"
    assert normalize_module_name('module') == ''
    assert normalize_module_name('no.parallel.modules') == 'no.parallel.modules'
    name_test('imagenet', 'vgg19')
    name_test('cifar10', 'resnet20_cifar')
    name_test('imagenet', 'alexnet')


@pytest.mark.parametrize('dataset, arch', [('imagenet', 'vgg19'),
                                           ('cifar10', 'resnet20_cifar'),
                                           ('imagenet', 'alexnet'),
                                           ('imagenet', 'resnext101_32x4d')])
def test_named_params_layers(dataset, arch, parallel):
    model = create_model(False, dataset, arch, parallel=parallel)
    sgraph = SummaryGraph(model, distiller.get_dummy_input(dataset))
    sgraph_layer_names = set(k for k, i, j in sgraph.named_params_layers())
    for layer_name in sgraph_layer_names:
        assert sgraph.find_op(layer_name) is not None, '{} was not found in summary graph'.format(layer_name)


def test_onnx_name_2_pytorch_name():
    assert onnx_name_2_pytorch_name("ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]") == "layer3.0.relu"
    assert onnx_name_2_pytorch_name('VGG/[features]/Sequential/Conv2d[34]') == "features.34"
    assert onnx_name_2_pytorch_name('NameWithNoModule') == ''


def test_connectivity_summary():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    summary = connectivity_summary(g)
    assert len(summary) == 80

    verbose_summary = connectivity_summary_verbose(g)
    assert len(verbose_summary) == 80


def test_sg_macs():
    '''Compare the MACs of different modules as computed by a SummaryGraph
    and model summary.'''
    import common
    sg = create_graph('imagenet', 'mobilenet')
    assert sg
    model, _ = common.setup_test('mobilenet', 'imagenet', parallel=False)
    df_compute = distiller.model_performance_summary(model, distiller.get_dummy_input('imagenet'))
    modules_macs = df_compute.loc[:, ['Name', 'MACs']]
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            summary_macs = int(modules_macs.loc[modules_macs.Name == name].MACs)
            sg_macs = sg.find_op(name)['attrs']['MACs']
            assert summary_macs == sg_macs


@pytest.mark.parametrize('dataset, arch', [('cifar10', 'resnet20_cifar'),
                                           ('imagenet', 'alexnet'),
                                           ('imagenet', 'resnext101_32x4d')])
def test_weights_size_attr(dataset, arch, parallel):
    model = create_model(False, dataset, arch, parallel=parallel)
    sgraph = SummaryGraph(model, distiller.get_dummy_input(dataset))

    distiller.assign_layer_fq_names(model)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            op = sgraph.find_op(name)
            assert op is not None
            assert op['attrs']['weights_vol'] == distiller.volume(mod.weight)


def test_merge_pad_avgpool():
    class ModelWithAvgPool(nn.Module):
        def __init__(self):
            super(ModelWithAvgPool, self).__init__()
            self.conv = nn.Conv2d(3, 10, 5)
            self.avgpool = nn.AvgPool2d(2)

        def forward(self, input):
            return self.avgpool(self.conv(input))

    m = ModelWithAvgPool()
    sg = SummaryGraph(m, distiller.get_dummy_input(input_shape=(1, 3, 50, 50)))

    avgpool_ops = [op_name for op_name in sg.ops if 'avgpool' in op_name]
    assert len(avgpool_ops) == 1
    assert sg.ops[avgpool_ops[0]]['name'] == 'avgpool'
    assert sg.ops[avgpool_ops[0]]['type'] == 'AveragePool'


def test_gemm_nodes_scope_names():
    class ModelWithGemms(nn.Module):
        def __init__(self):
            super(ModelWithGemms, self).__init__()
            self.drop1 = nn.Dropout()
            self.fc1 = nn.Linear(100, 50)
            self.relu1 = nn.ReLU(inplace=True)
            self.drop2 = nn.Dropout()
            self.fc2 = nn.Linear(50, 25)
            self.relu2 = nn.ReLU(inplace=True)
            self.fc3 = nn.Linear(25, 1)

        def forward(self, x):
            # Isn't this pretty...
            return self.fc3(self.relu2(self.fc2(self.drop2(self.relu1(self.fc1(self.drop1(x)))))))

    m = ModelWithGemms()
    sg = SummaryGraph(m, distiller.get_dummy_input(input_shape=(1, 100)))

    # For the model above we expect the ops to be named (in order):
    #   'drop1', 'fc1', 'relu1', 'drop2', 'fc2', 'relu2', 'fc3'
    # But without our workaround in place, they'll be named:
    #   'drop1', 'drop1__1', 'relu1', 'drop2', 'drop2__1', 'relu2', 'relu2__1'
    # (that is - each FC node gets the name of the node before)
    names, types = zip(*[(op_name, op['type']) for op_name, op in sg.ops.items()])
    assert names == ('drop1', 'fc1', 'relu1', 'drop2', 'fc2', 'relu2', 'fc3')
    assert types == ('Dropout', 'Gemm', 'Relu', 'Dropout', 'Gemm', 'Relu', 'Gemm')


@pytest.fixture(params=[False, True], ids=['dedicated_modules_off', 'dedicated_modules_on'])
def dedicated_modules(request):
    return request.param


def test_adjacency_map(parallel, dedicated_modules):
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv = nn.Conv2d(3, 10, 5)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU()

        def forward(self, x):
            res = self.conv(x)
            y = self.bn(res)
            y = self.relu(y)
            return y + res

    def check_adj_entry(actual, expected):
        assert actual.op_meta == expected.op_meta
        assert actual.predecessors == expected.predecessors
        assert actual.successors == expected.successors

    prefix = 'module.' if parallel else ''

    m = TestModel()
    if parallel:
        m = nn.DataParallel(m)
    sg = SummaryGraph(m, distiller.get_dummy_input(input_shape=(1, 3, 10, 10)))
    adj_map = sg.adjacency_map(dedicated_modules_only=dedicated_modules)

    if dedicated_modules:
        assert len(adj_map) == 3
    else:
        assert len(adj_map) == 4

    conv_op_meta = OpSimpleMetadata(prefix + 'conv', 'Conv')
    bn_op_meta = OpSimpleMetadata(prefix + 'bn', 'BatchNormalization')
    relu_op_meta = OpSimpleMetadata(prefix + 'relu', 'Relu')
    add_op_meta = OpSimpleMetadata('top_level_op', 'Add')

    name = conv_op_meta.name
    assert name in adj_map
    expected = AdjacentsEntry(conv_op_meta)
    expected.successors = [bn_op_meta] if dedicated_modules else [bn_op_meta, add_op_meta]
    check_adj_entry(adj_map[name], expected)

    name = bn_op_meta.name
    assert name in adj_map
    expected = AdjacentsEntry(bn_op_meta)
    expected.predecessors = [conv_op_meta]
    expected.successors = [relu_op_meta]
    check_adj_entry(adj_map[name], expected)

    name = relu_op_meta.name
    assert name in adj_map
    expected = AdjacentsEntry(relu_op_meta)
    expected.predecessors = [bn_op_meta]
    expected.successors = [] if dedicated_modules else [add_op_meta]
    check_adj_entry(adj_map[name], expected)

    name = add_op_meta.name
    if dedicated_modules:
        assert name not in adj_map
    else:
        assert name in adj_map
        expected = AdjacentsEntry(add_op_meta)
        expected.predecessors = [relu_op_meta, conv_op_meta]
        check_adj_entry(adj_map[name], expected)


if __name__ == '__main__':
    #test_connectivity_summary()
    test_sg_macs()