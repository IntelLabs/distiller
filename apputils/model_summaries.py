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

"""This module provides various model summary information.

This code is currently in "experimental" state, as we wait for the next release
of PyTorch with stable support for the JIT tracer functionality we employ in this
code (it was built with a 4.x master branch).
"""

import re
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.jit as jit
import pandas as pd
from tabulate import tabulate

def onnx_name_2_pytorch_name(name):
    # Convert a layer's name from an ONNX name, to a PyTorch name
    # For example:
    #   ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu].1 ==> layer3.0.relu.1

    # First see if there's an instance identifier
    instance = ''
    if name.find('.')>0:
        instance = name[name.find('.')+1 :]

    # Next, split by square brackets
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]

    #print(op['name'] , '.'.join(name_parts), op['type'])
    name = '.'.join(name_parts) + instance
    #if name == '':
    #    name += 'stam'
    return name


class SummaryGraph(object):
    """We use Pytorch's JIT tracer to run a forward pass and generate a trace graph, which
    is an internal representation of the model.  We then use ONNX to "clean" this
    representation.  After builiding a new representation of the graph, we can print
    it to a table, a PNG, add information to nodes, etc.

    The trace is a C++ component and the API is not documented, so we need to dig into some
    Pytorch internals code to understand how to get the info we need.
    https://github.com/pytorch/pytorch/blob/master/torch/onnx/__init__.py
    https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py

    We think that using the trace output to generate a representation of the graph, is
    the best method available in Pytorch, due to the module's dynamic nature.
    Pytorch's module API naturally ignores layers/operations which are implemented as
    torch.autograd.Function, without an nn.Module.  For example:
        out = F.relu(self.bn1(self.conv1(x)))

    Another case where traversing the nn.Module API is not sufficient to create a
    representation of the graph, is the same nn.Module is used several times in the
    graph.  For example:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)    <=== First use of self.relu

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)    <=== Second use of self.relu
    """
    def __init__(self, model, dummy_input):
        with torch.onnx.set_training(model, False):
            trace, _ = jit.get_trace_graph(model, dummy_input)

            # Let ONNX do the heavy lifting: fusing the convolution nodes; fusing the nodes
            # composing a GEMM operation; etc.
            torch.onnx._optimize_trace(trace, False)
            graph = trace.graph()
            self.ops = []
            self.params = {}
            self.edges = []
            self.temp = {}

            in_out = list(graph.inputs()) + list(graph.outputs())
            for param in in_out:
                self.__add_param(param)

            for node in graph.nodes():
                op = {}
                op['name'] = node.scopeName()
                op['orig-name'] = node.scopeName()
                op['type'] = node.kind().lstrip('::onnx')
                op['inputs'] = []
                op['outputs'] = []
                op['params'] = []

                # in-place operators create very confusing graphs (Resnet, for example),
                # so we "unroll" them
                same = [layer for layer in self.ops if layer['orig-name'] == op['orig-name']]
                if len(same) > 0:
                    op['name'] += "." + str(len(same))

                op['name'] = onnx_name_2_pytorch_name(op['name'])
                #op['name'] = '\n'.join(op['name'], op['type'])
                op['name'] += ("\n" + op['type'])

                # print(op['name'])
                self.ops.append(op)

                for input_ in node.inputs():
                    self.__add_input(op, input_)
                    self.edges.append((input_.uniqueName(), op['name']))

                for output in node.outputs():
                    self.__add_output(op, output)
                    self.edges.append((op['name'], output.uniqueName()))

                op['attrs'] = {attr_name: node[attr_name] for attr_name in node.attributeNames()}

        self.add_macs_attr()
        self.add_footprint_attr()
        self.add_arithmetic_intensity_attr()


    def __add_input(self, op, n):
        param = self.__add_param(n)
        if param is None: return
        if param['id'] not in op['inputs']:
            op['inputs'].append(param['id'])

    def __add_output(self, op, n):
        param = self.__add_param(n)
        if param is None: return
        if param['id'] not in op['outputs']:
            op['outputs'].append(param['id'])

    def __add_param(self, n):
        param = {}
        if n.uniqueName() not in self.params:
            param = self.__tensor_desc(n)
            self.params[n.uniqueName()] = param
        else:
            param = self.params[n.uniqueName()]
        return param

    def __tensor_desc(self, n):
        tensor = {}
        tensor['id'] = n.uniqueName()
        try:
            s = str(n.type())
            tensor['type'] = s[:s.find('(')]
            s = s[s.find('(')+1: s.find(')')]
            tensor['shape'] = tuple(map(lambda x: int(x), s.split(',')))
        except:
            return None
        return tensor

    def param_shape(self, param_id):
        return self.params[param_id]['shape']

    @staticmethod
    def volume(dims):
        return np.prod(dims)

    def param_volume(self, param_id):
        return SummaryGraph.volume(self.param_shape(param_id))

    def add_macs_attr(self):
        for op in self.ops:
            op['attrs']['MACs'] = 0
            if op['type'] == 'Conv':
                conv_out = op['outputs'][0]
                conv_in =  op['inputs'][0]
                conv_w = op['attrs']['kernel_shape']
                ofm_vol = self.param_volume(conv_out)
                # MACs = volume(OFM) * (#IFM * K^2)
                op['attrs']['MACs'] = ofm_vol * SummaryGraph.volume(conv_w) * self.params[conv_in]['shape'][1]
            elif op['type'] == 'Gemm':
                conv_out =  op['outputs'][0]
                conv_in =  op['inputs'][0]
                n_ifm = self.param_shape(conv_in)[1]
                n_ofm = self.param_shape(conv_out)[1]
                # MACs = #IFM * #OFM
                op['attrs']['MACs'] = n_ofm * n_ifm

    def add_footprint_attr(self):
        for op in self.ops:
            op['attrs']['footprint'] = 0
            if op['type'] in ['Conv', 'Gemm', 'MaxPool']:
                conv_out = op['outputs'][0]
                conv_in =  op['inputs'][0]
                ofm_vol = self.param_volume(conv_out)
                ifm_vol = self.param_volume(conv_in)
                if op['type'] == 'Conv' or op['type'] == 'Gemm':
                    conv_w = op['inputs'][1]
                    weights_vol = self.param_volume(conv_w)
                    #print(ofm_vol , ifm_vol , weights_vol)
                    op['attrs']['footprint'] = ofm_vol + ifm_vol + weights_vol
                    op['attrs']['fm_vol'] = ofm_vol + ifm_vol
                    op['attrs']['weights_vol'] = weights_vol
                elif op['type'] == 'MaxPool':
                    op['attrs']['footprint'] = ofm_vol + ifm_vol

    def add_arithmetic_intensity_attr(self):
        for op in self.ops:
            if op['attrs']['footprint'] == 0:
                op['attrs']['ai'] = 0
            else:
                # integers are enough, and note that we also round up
                op['attrs']['ai'] = ((op['attrs']['MACs']+0.5*op['attrs']['footprint']) // op['attrs']['footprint'])

    def get_attr(self, attr, f = lambda op: True):
        return [op['attrs'][attr] for op in self.ops if attr in op['attrs'] and f(op)]

    def get_ops(self, attr, f = lambda op: True):
        return [op for op in self.ops if attr in op['attrs'] and f(op)]


def attributes_summary(sgraph, ignore_attrs):
    """Generate a summary of a graph's attributes.

    Args:
        sgraph: a SummaryGraph instance
        ignore_attrs: a list of attributes to ignore in the output datafraem

    Output:
        A Pandas dataframe
    """
    def pretty_val(val):
        if type(val) == int:
            return format(val, ",d")
        return str(val)

    def pretty_attrs(attrs, ignore_attrs):
        ret = ''
        for key, val in attrs.items():
            if key in ignore_attrs:
                continue
            ret += key + ': ' + pretty_val(val) + '\n'
        return ret

    df = pd.DataFrame(columns=['Name', 'Type', 'Attributes'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops):
        df.loc[i] = [op['name'], op['type'], pretty_attrs(op['attrs'], ignore_attrs)]
    return df

def attributes_summary_tbl(sgraph, ignore_attrs):
    df = attributes_summary(sgraph, ignore_attrs)
    return tabulate(df, headers='keys', tablefmt='psql')

def connectivity_summary(sgraph):
    """Generate a summary of each node's connectivity.

    Args:
        sgraph: a SummaryGraph instance
    """
    df = pd.DataFrame(columns=['Name', 'Type', 'Inputs', 'Outputs'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops):
        df.loc[i] = [op['name'], op['type'], op['inputs'], op['outputs']]
    return df

def connectivity_summary_verbose(sgraph):
    """Generate a summary of each node's connectivity, with details
    about the parameters.

    Args:
        sgraph: a SummaryGraph instance
    """
    def format_list(l):
        ret = ''
        for i in l: ret += str(i) + '\n'
        return ret[:-1]

    df = pd.DataFrame(columns=['Name', 'Type', 'Inputs', 'Outputs'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops):
        outputs = []
        for blob in op['outputs']:
            if blob in sgraph.params:
                outputs.append(blob + ": " + str(sgraph.params[blob]['shape']))
        inputs = []
        for blob in op['inputs']:
            if blob in sgraph.params:
                inputs.append(blob + ": " + str(sgraph.params[blob]['shape']))
        inputs = format_list(inputs)
        outputs = format_list(outputs)
        #t.add_row([op['name'], op['type'], inputs, outputs])
        df.loc[i] = [op['name'], op['type'], inputs, outputs]

    return df

def connectivity_tbl_summary(sgraph, verbose=False):
    if verbose:
        df = connectivity_summary_verbose(sgraph)
    else:
        df = connectivity_summary(sgraph)
    return tabulate(df, headers='keys', tablefmt='psql')



import pydot

def create_pydot_graph(op_nodes, data_nodes, param_nodes, edges):
    pydot_graph = pydot.Dot('Net', graph_type='digraph', rankdir='TB')

    node_style = {'shape': 'record',
                  'fillcolor': '#6495ED',
                  'style': 'rounded, filled'}

    for op_node in op_nodes:
        pydot_graph.add_node(pydot.Node(op_node, **node_style))

    for data_node in data_nodes:
        pydot_graph.add_node(pydot.Node(data_node))

    node_style = {'shape': 'oval',
                  'fillcolor': 'gray',
                  'style': 'rounded, filled'}

    if param_nodes is not None:
        for data_node in param_nodes:
            pydot_graph.add_node(pydot.Node(data_node, **node_style))

    for edge in edges:
        pydot_graph.add_edge(pydot.Edge(edge[0], edge[1]))

    return pydot_graph

def draw_model_to_file(sgraph, png_fname):
    """Create a PNG file, containing a graphiz-dot graph of the netowrk represented
    by SummaryGraph 'sgraph'
    """
    png = create_png(sgraph)
    with open(png_fname, 'wb') as fid:
        fid.write(png)

def draw_img_classifier_to_file(model, png_fname, dataset):
    try:
        if dataset == 'imagenet':
            dummy_input = Variable(torch.randn(1, 3, 224, 224), requires_grad=False)
        elif dataset == 'cifar10':
            dummy_input = Variable(torch.randn(1, 3, 32, 32))
        else:
            print("Unsupported dataset (%s) - aborting draw operation" % dataset)
            return

        g = SummaryGraph(model, dummy_input)
        draw_model_to_file(g, png_fname)
        print("Network PNG image generation completed")
    except TypeError as e:
        print("An error has occured while generating the network PNG image.")
        print("This feature is not supported on official PyTorch releases.")
        print("Please check that you are using a valid PyTorch version.")
        print("You are using pytorch version %s" %torch.__version__)
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz")

def create_png(sgraph):
    """Create a PNG object containing a graphiz-dot graph of the netowrk represented
    by SummaryGraph 'sgraph'
    """
    op_nodes = [op['name'] for op in sgraph.ops]
    data_nodes = [id for id in sgraph.params.keys() if data_node_has_parent(sgraph, id)]
    param_nodes = [id for id in sgraph.params.keys() if not data_node_has_parent(sgraph, id)]
    edges = sgraph.edges

    display_param_node = False
    if not display_param_node:
        edges = [edge for edge in sgraph.edges if edge[0] in (data_nodes+op_nodes)]
        param_nodes = None

    if False:
        data_nodes = None

    pydot_graph = create_pydot_graph(op_nodes, data_nodes, param_nodes, edges)
    png = pydot_graph.create_png()
    return png

def data_node_has_parent(g, id):
    for edge in g.edges:
        if edge[1] == id: return True
    return False
