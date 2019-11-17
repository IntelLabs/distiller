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
from copy import deepcopy
from collections import OrderedDict
import distiller
from distiller.models import ALL_MODEL_NAMES, create_model
from distiller.apputils import *
from distiller import normalize_module_name, denormalize_module_name, \
    SummaryGraph, onnx_name_2_pytorch_name
from distiller.model_summaries import connectivity_summary, connectivity_summary_verbose
from distiller.summary_graph import AdjacentsEntry, OpSimpleMetadata, _DistillerModuleList, \
    _to_distiller_modulelist, _named_modules_with_duplicates

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
    assert edges[0].src == 'input.1' and edges[0].dst == 'conv1'

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
    assert preds == ['input.1', '1']


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
    assert preds == prefix_strs(['conv1', 'layer1.0.conv2'], prefix)


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


def test_scope_name_workarounds():
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
            self.drop3 = nn.Dropout()

        def forward(self, x):
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.drop3(x)
            return x

    m = ModelWithGemms()
    dummy_input = distiller.get_dummy_input(input_shape=(1, 100))
    expected_types = ('Gemm', 'Relu', 'Gemm', 'Relu', 'Gemm')

    # We have a workaround for the following issue:
    # (used to be 2 issues but one got fixed in PyTorch 1.2)
    #   * Ops that come before a dropout op get the scope name of the dropout op

    # For the model above we expect the ops in the graph to be named (in order):
    #   'fc1', 'relu1', 'fc2', 'relu2', 'fc3'
    # (note that dropout ops are dropped)
    #
    # But since 'relu1' and 'fc3' come before a dropout op, without the workaround in place we'll get:
    #   'fc1', 'drop2', 'fc2', 'relu2', 'drop3'

    # We test without the workarounds as a means to see if the issues still exist. New PyTorch versions
    # may fix them, in which case we can remove the workarounds
    sg = SummaryGraph(m, dummy_input, apply_scope_name_workarounds=False)
    names, types = zip(*[(op_name, op['type']) for op_name, op in sg.ops.items()])
    assert names == ('fc1', 'drop2', 'fc2', 'relu2', 'drop3')
    assert types == expected_types

    # Now test with the workarounds
    sg = SummaryGraph(m, dummy_input)
    names, types = zip(*[(op_name, op['type']) for op_name, op in sg.ops.items()])
    assert names == ('fc1', 'relu1', 'fc2', 'relu2', 'fc3')
    assert types == expected_types


@pytest.fixture(params=[False, True], ids=['dedicated_modules_off', 'dedicated_modules_on'])
def dedicated_modules(request):
    return request.param


def test_adjacency_map(parallel, dedicated_modules):
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv = nn.Conv2d(3, 10, 5)
            self.bn = nn.BatchNorm2d(10)
            self.post_conv_bn = nn.ModuleList([
                nn.Tanh(),
                nn.ReLU()
            ])

        def forward(self, x):
            res = self.conv(x)
            y = self.bn(res)
            for m in self.post_conv_bn:
                y = m(y)
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
        assert len(adj_map) == 4
    else:
        assert len(adj_map) == 5

    conv_op_meta = OpSimpleMetadata(prefix + 'conv', 'Conv')
    bn_op_meta = OpSimpleMetadata(prefix + 'bn', 'BatchNormalization')
    tanh_op_meta = OpSimpleMetadata(prefix + 'post_conv_bn.0', 'Tanh')
    relu_op_meta = OpSimpleMetadata(prefix + 'post_conv_bn.1', 'Relu')
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
    expected.successors = [tanh_op_meta]
    check_adj_entry(adj_map[name], expected)

    name = tanh_op_meta.name
    assert name in adj_map
    expected = AdjacentsEntry(tanh_op_meta)
    expected.predecessors = [bn_op_meta]
    expected.successors = [relu_op_meta]
    check_adj_entry(adj_map[name], expected)

    name = relu_op_meta.name
    assert name in adj_map
    expected = AdjacentsEntry(relu_op_meta)
    expected.predecessors = [tanh_op_meta]
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


#############################################################
# Conversion to DistillerModuleList tests
#############################################################

# Model for testing conversion of nested ModuleLists
class ListsModule(nn.Module):
    def __init__(self):
        super(ListsModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.post_conv1 = nn.ModuleList([
            nn.BatchNorm2d(10),
            nn.ModuleList([
                nn.ReLU(),
                nn.ModuleList([nn.Tanh(), nn.MaxPool2d(2)])])])
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.post_conv2 = nn.ModuleList([nn.ReLU6(), nn.MaxPool2d(4)])

        self.expected_mlist_to_dmlist = OrderedDict([
            ('post_conv1', ['post_conv1']),
            ('post_conv1.1', ['post_conv1', '1']),
            ('post_conv1.1.1', ['post_conv1', '1', '1']),
            ('post_conv2', ['post_conv2']),
        ])
        self.expected_list_contents_name_changes = OrderedDict([
            ('post_conv1.0', 'post_conv1_0'),
            ('post_conv1.1.0', 'post_conv1_1_0'),
            ('post_conv1.1.1.0', 'post_conv1_1_1_0'),
            ('post_conv1.1.1.1', 'post_conv1_1_1_1'),
            ('post_conv2.0', 'post_conv2_0'),
            ('post_conv2.1', 'post_conv2_1'),
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.post_conv1[0](x)
        x = self.post_conv1[1][0](x)
        for m in self.post_conv1[1][1]:
            x = m(x)
        x = self.conv2(x)
        for m in self.post_conv2:
            x = m(x)
        return x


# Model for testing conversion in case of nested containers
class Block(nn.Module):
    def __init__(self, in_ch):
        super(Block, self).__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch * 2
        self.conv = nn.Conv2d(in_ch, self.out_ch, 3)
        self.post_conv = nn.ModuleList([nn.BatchNorm2d(self.out_ch), nn.ReLU()])

    def forward(self, x):
        x = self.conv(x)
        for m in self.post_conv:
            x = m(x)
        return x


class BlocksModule(nn.Module):
    def __init__(self):
        super(BlocksModule, self).__init__()
        self.block1 = Block(3)
        self.blocks2_3 = nn.Sequential(Block(6), Block(12))
        self.blocks4_5 = nn.ModuleList([Block(24), Block(48)])
        self.block6 = Block(96)

        self.expected_mlist_to_dmlist = OrderedDict([
            ('block1.post_conv', ['block1', 'post_conv']),
            ('blocks2_3.0.post_conv', ['blocks2_3', '0', 'post_conv']),
            ('blocks2_3.1.post_conv', ['blocks2_3', '1', 'post_conv']),
            ('blocks4_5', ['blocks4_5']),
            ('blocks4_5.0.post_conv', ['blocks4_5', '0', 'post_conv']),
            ('blocks4_5.1.post_conv', ['blocks4_5', '1', 'post_conv']),
            ('block6.post_conv', ['block6', 'post_conv']),
        ])
        self.expected_list_contents_name_changes = OrderedDict([
            ('block1.post_conv.0', 'block1.post_conv_0'),
            ('block1.post_conv.1', 'block1.post_conv_1'),
            ('blocks2_3.0.post_conv.0', 'blocks2_3.0.post_conv_0'),
            ('blocks2_3.0.post_conv.1', 'blocks2_3.0.post_conv_1'),
            ('blocks2_3.1.post_conv.0', 'blocks2_3.1.post_conv_0'),
            ('blocks2_3.1.post_conv.1', 'blocks2_3.1.post_conv_1'),
            ('blocks4_5.0', 'blocks4_5_0'),
            ('blocks4_5.0.conv', 'blocks4_5_0.conv'),
            ('blocks4_5.0.post_conv.0', 'blocks4_5_0.post_conv_0'),
            ('blocks4_5.0.post_conv.1', 'blocks4_5_0.post_conv_1'),
            ('blocks4_5.1', 'blocks4_5_1'),
            ('blocks4_5.1.conv', 'blocks4_5_1.conv'),
            ('blocks4_5.1.post_conv.0', 'blocks4_5_1.post_conv_0'),
            ('blocks4_5.1.post_conv.1', 'blocks4_5_1.post_conv_1'),
            ('block6.post_conv.0', 'block6.post_conv_0'),
            ('block6.post_conv.1', 'block6.post_conv_1'),
        ])

    def forward(self, x):
        x = self.block1(x)
        x = self.blocks2_3(x)
        for block in self.blocks4_5:
            x = block(x)
        x = self.block6(x)
        return x


# Model with duplicate modules
class ModelWithDuplicates(nn.Module):
    def __init__(self):
        super(ModelWithDuplicates, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.post_conv1 = nn.ModuleList([nn.ReLU(), nn.Tanh()])
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.post_conv2 = self.post_conv1

        self.expected_mlist_to_dmlist = OrderedDict([
            ('post_conv1', ['post_conv1']),
            ('post_conv2', ['post_conv2']),
        ])
        self.expected_list_contents_name_changes = OrderedDict([
            ('post_conv1.0', 'post_conv1_0'),
            ('post_conv1.1', 'post_conv1_1'),
            ('post_conv2.0', 'post_conv2_0'),
            ('post_conv2.1', 'post_conv2_1'),
        ])

    def forward(self, x):
        x = self.conv1(x)
        for m in self.post_conv1:
            x = m(x)
        x = self.conv2(x)
        for m in self.post_conv2:
            x = m(x)
        return x


@pytest.mark.parametrize("model", [ListsModule(), BlocksModule(), ModelWithDuplicates()],
                         ids=['ListsModule', 'BlocksModule', 'ModelWithDuplicates'])
def test_distiller_module_list_conversion(model):
    def check_equal_tensors(actual, expected):
        assert (actual == expected).all().item() == 1

    model_dml, converted_module_names_map = _to_distiller_modulelist(deepcopy(model))

    # Check all modules converted as expected
    named_modules_dmlist = OrderedDict(_named_modules_with_duplicates(model_dml))
    for name_orig, module_orig in _named_modules_with_duplicates(model):
        if name_orig in model.expected_mlist_to_dmlist:
            # Check ModuleLists were converted to an attribute with the expected name, which is not
            # registered as a module in the converted model
            assert name_orig not in named_modules_dmlist
            attr_dml = model_dml
            for attr_name in model.expected_mlist_to_dmlist[name_orig]:
                try:
                    attr_dml = attr_dml[int(attr_name)]
                except ValueError:
                    attr_dml = getattr(attr_dml, attr_name)
            assert isinstance(attr_dml, _DistillerModuleList)
        else:
            # Check module name changed as expected, and that the module type didn't change
            expected_name_dml = model.expected_list_contents_name_changes.get(name_orig, name_orig)
            assert expected_name_dml in named_modules_dmlist
            assert expected_name_dml in converted_module_names_map
            assert converted_module_names_map[expected_name_dml] == name_orig
            assert type(named_modules_dmlist[expected_name_dml]) == type(module_orig)
            converted_module_names_map.pop(expected_name_dml)
            named_modules_dmlist.pop(expected_name_dml)

    assert not converted_module_names_map, 'Unexpected contents in converted_module_names_map'
    assert not named_modules_dmlist, 'Unexpected contents in converted model named_modules'

    # Now make sure all parameters and buffers didn't change
    for p_orig, p_dml in zip(model.parameters(), model_dml.parameters()):
        check_equal_tensors(p_dml, p_orig)
    for b_orig, b_dml in zip(model.buffers(), model_dml.buffers()):
        check_equal_tensors(b_dml, b_orig)

    # Check forward pass gives identical results
    x = torch.randn(1, 3, 50, 50)
    y_orig = model(x)
    y_dml = model_dml(x)
    check_equal_tensors(y_dml, y_orig)


if __name__ == '__main__':
    #test_connectivity_summary()
    test_sg_macs()