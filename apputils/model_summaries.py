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

This code is proven to work on CNN image classification models using PyTorch 04.
RNNs are currently not working well.
"""

import os
import re
import numpy as np
import collections
from copy import deepcopy
import torch
import torchvision
from torch.autograd import Variable
import torch.jit as jit
import pandas as pd
from tabulate import tabulate
import pydot
import distiller
import logging
msglogger = logging.getLogger(__name__)


def onnx_name_2_pytorch_name(name, op_type):
    # Convert a layer's name from an ONNX name, to a PyTorch name
    # For example:
    #   ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu].1 ==> layer3.0.relu.1

    # First see if there's an instance identifier
    instance = ''
    if name.find('.') > 0:
        instance = name[name.find('.')+1:]

    # Next, split by square brackets
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]

    # If name doesn't have the pattern above, it probably means the op was called via
    # some functional API and not via a module. Couple of examples:
    #   x = x.view(...)
    #   x = F.relu(x)
    # In this case, to have a meaningful name, we use the op type
    new_name = ('.'.join(name_parts) if len(name_parts) > 0 else op_type) + instance

    msglogger.debug("new sgraph node {} {} {}".format(name, op_type, new_name))
    return new_name


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
    Edge = collections.namedtuple('Edge', 'src dst')

    def __init__(self, model, dummy_input):
        with torch.onnx.set_training(model, False):
            trace, _ = jit.get_trace_graph(model, dummy_input.cuda())

            # Let ONNX do the heavy lifting: fusing the convolution nodes; fusing the nodes
            # composing a GEMM operation; etc.
            torch.onnx._optimize_trace(trace, False)
            graph = trace.graph()
            self.ops = {}
            self.params = {}
            self.edges = []
            self.temp = {}

            in_out = list(graph.inputs()) + list(graph.outputs())
            for param in in_out:
                self.__add_param(param)

            for node in graph.nodes():
                new_op = self.__create_op(node)

                # Operators with the same name create very confusing graphs (Resnet, for example),
                # so we "unroll" them.
                # Sometimes operations of different types have the same name, so we differentiate
                # using both name and type
                # (this happens, for example, when an operator is called via some functional API and
                # not via a module)
                same = [op for op in self.ops.values() if
                        op['orig-name'] + op['type'] == new_op['orig-name'] + new_op['type']]
                if len(same) > 0:
                    new_op['name'] += "." + str(len(same))

                new_op['name'] = onnx_name_2_pytorch_name(new_op['name'], new_op['type'])
                assert len(new_op['name']) > 0

                self.ops[new_op['name']] = new_op

                for input_ in node.inputs():
                    self.__add_input(new_op, input_)
                    self.edges.append(SummaryGraph.Edge(input_.uniqueName(), new_op['name']))

                for output in node.outputs():
                    self.__add_output(new_op, output)
                    self.edges.append(SummaryGraph.Edge(new_op['name'], output.uniqueName()))

                new_op['attrs'] = {attr_name: node[attr_name] for attr_name in node.attributeNames()}

        self.add_macs_attr()
        self.add_footprint_attr()
        self.add_arithmetic_intensity_attr()

    def __create_op(self, onnx_node):
        op = {}
        op['name'] = onnx_node.scopeName()
        op['orig-name'] = onnx_node.scopeName()
        op['type'] = onnx_node.kind().lstrip('::onnx')
        op['inputs'] = []
        op['outputs'] = []
        op['params'] = []
        return op

    def __add_input(self, op, n):
        param = self.__add_param(n)
        if param is None:
            return
        if param['id'] not in op['inputs']:
            op['inputs'].append(param['id'])

    def __add_output(self, op, n):
        param = self.__add_param(n)
        if param is None:
            return
        if param['id'] not in op['outputs']:
            op['outputs'].append(param['id'])

    def __add_param(self, n):
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
            # try parsing the FM tensor type.  For example: Float(1, 64, 8, 8)
            s = str(n.type())
            s = s[s.find('(')+1: s.find(')')]
            tensor['shape'] = tuple(map(lambda x: int(x), s.split(',')))
        except ValueError:
            # Size not specified in type
            tensor['shape'] = 0,
        return tensor

    def param_shape(self, param_id):
        return self.params[param_id]['shape']

    @staticmethod
    def volume(dims):
        return np.prod(dims)

    def param_volume(self, param_id):
        return SummaryGraph.volume(self.param_shape(param_id))

    def add_macs_attr(self):
        for op in self.ops.values():
            op['attrs']['MACs'] = 0
            if op['type'] == 'Conv':
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                conv_w = op['attrs']['kernel_shape']
                ofm_vol = self.param_volume(conv_out)
                # MACs = volume(OFM) * (#IFM * K^2)
                op['attrs']['MACs'] = ofm_vol * SummaryGraph.volume(conv_w) * self.params[conv_in]['shape'][1]
            elif op['type'] == 'Gemm':
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                n_ifm = self.param_shape(conv_in)[1]
                n_ofm = self.param_shape(conv_out)[1]
                # MACs = #IFM * #OFM
                op['attrs']['MACs'] = n_ofm * n_ifm

    def add_footprint_attr(self):
        for op in self.ops.values():
            op['attrs']['footprint'] = 0
            if op['type'] in ['Conv', 'Gemm', 'MaxPool']:
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                ofm_vol = self.param_volume(conv_out)
                ifm_vol = self.param_volume(conv_in)
                if op['type'] == 'Conv' or op['type'] == 'Gemm':
                    conv_w = op['inputs'][1]
                    weights_vol = self.param_volume(conv_w)
                    op['attrs']['footprint'] = ofm_vol + ifm_vol + weights_vol
                    op['attrs']['fm_vol'] = ofm_vol + ifm_vol
                    op['attrs']['weights_vol'] = weights_vol
                elif op['type'] == 'MaxPool':
                    op['attrs']['footprint'] = ofm_vol + ifm_vol

    def add_arithmetic_intensity_attr(self):
        for op in self.ops.values():
            if op['attrs']['footprint'] == 0:
                op['attrs']['ai'] = 0
            else:
                # integers are enough, and note that we also round up
                op['attrs']['ai'] = ((op['attrs']['MACs']+0.5*op['attrs']['footprint']) // op['attrs']['footprint'])

    def get_attr(self, attr, f=lambda op: True):
        return [op['attrs'][attr] for op in self.ops.values() if attr in op['attrs'] and f(op)]

    def get_ops(self, attr, f=lambda op: True):
        return [op for op in self.ops.values() if attr in op['attrs'] and f(op)]

    def find_op(self, lost_op_name):
        assert isinstance(lost_op_name, str)
        return self.ops.get(lost_op_name, None)

    def find_param(self, data_name):
        return self.params.get(data_name, None)

    def predecessors(self, op, depth, done_list=None):
        """Returns a list of <op>'s predecessors"""

        if done_list is None:
            done_list = []

        if isinstance(op, dict):
            preds = [edge.src for edge in self.edges if (edge.dst == op['name'] and
                                                         edge.src not in done_list)]
            done_list += preds
        else:
            preds = [edge.src for edge in self.edges if (edge.dst == op and
                                                         edge.src not in done_list)]
            done_list += preds

        if depth == 1:
            return preds
        else:
            ret = []
            for predecessor in preds:
                ret += self.predecessors(predecessor, depth-1, done_list)
            return ret

    def predecessors_f(self, node_name, predecessors_types, done_list=None, logging=None):
        """Returns a list of <op>'s predecessors, if they match the <predecessors_types> criteria.
        """
        node = self.find_op(node_name)
        node_is_an_op = True
        if node is None:
            node_is_an_op = False
            node = self.find_param(node_name)
            if node is None:
                msglogger.warn("predecessors_f: Could not find node {}".format(node_name))
                return []

        if done_list is None:
            done_list = []

        done_list.append(node_name)

        if not isinstance(predecessors_types, list):
            predecessors_types = [predecessors_types]

        if node_is_an_op:
            # We check if we found the type of node we're looking for,
            # and that this is not the first node in our search.
            if node['type'] in predecessors_types and len(done_list) > 1:
                return [node_name]

            # This is an operation node
            preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                         edge.src not in done_list)]
        else:
            # This is a data node
            preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                         edge.src not in done_list)]
        ret = []
        for predecessor in preds:
            ret += self.predecessors_f(predecessor, predecessors_types, done_list, logging)
        return ret

    def successors(self, node, depth, done_list=None):
        """Returns a list of <op>'s successors"""

        if done_list is None:
            done_list = []

        if isinstance(node, dict):
            # This is an operation node
            succs = [edge.dst for edge in self.edges if (edge.src == node['name'] and
                                                         edge.dst not in done_list)]
            done_list += succs
        else:
            # This is a data node
            succs = [edge.dst for edge in self.edges if (edge.src == node and
                                                         edge.dst not in done_list)]
            done_list += succs

        if depth == 1:
            return succs
        else:
            ret = []
            for successor in succs:
                ret += self.successors(successor, depth-1, done_list)
            return ret

    def successors_f(self, node_name, successors_types, done_list=None, logging=None):
        """Returns a list of <op>'s successors, if they match the <successors_types> criteria.

        Traverse the graph, starting at node <node_name>, and search for successor
        nodes, that have one of the node types listed in <successors_types>.
        If none is found, then return an empty list.

        <node_name> and the returned list of successors are strings, because
        """

        node = self.find_op(node_name)
        node_is_an_op = True
        if node is None:
            node_is_an_op = False
            node = self.find_param(node_name)
            if node is None:
                #raise ValueError("something went wrong")
                return []

        if done_list is None:
            done_list = []

        done_list.append(node_name)

        if not isinstance(successors_types, list):
            successors_types = [successors_types]

        if node_is_an_op:
            # We check if we found the type of node we're looking for,
            # and that this is not the first node in our search.
            if node['type'] in successors_types and len(done_list) > 1:
                return [node_name]

            # This is an operation node
            succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                         edge.dst not in done_list)]
        else:
            # This is a data node
            succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                         edge.dst not in done_list)]
        ret = []
        for successor in succs:
            ret += self.successors_f(successor, successors_types, done_list, logging)
        return ret


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
    for i, op in enumerate(sgraph.ops.values()):
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
    for i, op in enumerate(sgraph.ops.values()):
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
        df.loc[i] = [op['name'], op['type'], inputs, outputs]

    return df


def connectivity_tbl_summary(sgraph, verbose=False):
    if verbose:
        df = connectivity_summary_verbose(sgraph)
    else:
        df = connectivity_summary(sgraph)
    return tabulate(df, headers='keys', tablefmt='psql')


def create_pydot_graph(op_nodes, data_nodes, param_nodes, edges, rankdir='TB', styles=None):
    """Low-level API to create a PyDot graph (dot formatted).
    """
    pydot_graph = pydot.Dot('Net', graph_type='digraph', rankdir=rankdir)

    op_node_style = {'shape': 'record',
                     'fillcolor': '#6495ED',
                     'style': 'rounded, filled'}

    for op_node in op_nodes:
        style = op_node_style
        # Check if we should override the style of this node.
        if styles is not None and op_node[0] in styles:
            style = styles[op_node[0]]
        pydot_graph.add_node(pydot.Node(op_node[0], **style, label="\n".join(op_node)))

    for data_node in data_nodes:
        pydot_graph.add_node(pydot.Node(data_node[0], label="\n".join(data_node[1:])))

    node_style = {'shape': 'oval',
                  'fillcolor': 'gray',
                  'style': 'rounded, filled'}

    if param_nodes is not None:
        for param_node in param_nodes:
            pydot_graph.add_node(pydot.Node(param_node[0], **node_style, label="\n".join(param_node[1:])))

    for edge in edges:
        pydot_graph.add_edge(pydot.Edge(edge[0], edge[1]))

    return pydot_graph


def create_png(sgraph, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG object containing a graphiz-dot graph of the network,
    as represented by SummaryGraph 'sgraph'.

    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    op_nodes = [op['name'] for op in sgraph.ops.values()]
    data_nodes = []
    param_nodes = []
    for id, param in sgraph.params.items():
        n_data = (id, str(distiller.volume(param['shape'])), str(param['shape']))
        if data_node_has_parent(sgraph, id):
            data_nodes.append(n_data)
        else:
            param_nodes.append(n_data)
    edges = sgraph.edges

    if not display_param_nodes:
        # Use only the edges that don't have a parameter source
        non_param_ids = op_nodes + [dn[0] for dn in data_nodes]
        edges = [edge for edge in sgraph.edges if edge.src in non_param_ids]
        param_nodes = None

    op_nodes = [(op['name'], op['type']) for op in sgraph.ops.values()]
    pydot_graph = create_pydot_graph(op_nodes, data_nodes, param_nodes, edges, rankdir, styles)
    png = pydot_graph.create_png()
    return png


def draw_model_to_file(sgraph, png_fname, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG file, containing a graphiz-dot graph of the netowrk represented
    by SummaryGraph 'sgraph'

    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        png_fname (string): PNG file name
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
        """
    png = create_png(sgraph, display_param_nodes=display_param_nodes)
    with open(png_fname, 'wb') as fid:
        fid.write(png)


def draw_img_classifier_to_file(model, png_fname, dataset, display_param_nodes=False,
                                rankdir='TB', styles=None):
    """Draw a PyTorch image classifier to a PNG file.  This a helper function that
    simplifies the interface of draw_model_to_file().

    Args:
        model: PyTorch model instance
        png_fname (string): PNG file name
        dataset (string): one of 'imagenet' or 'cifar10'.  This is required in order to
                          create a dummy input of the correct shape.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """
    try:
        dummy_input = dataset_dummy_input(dataset)
        model = distiller.make_non_parallel_copy(model)
        g = SummaryGraph(model, dummy_input)
        draw_model_to_file(g, png_fname, display_param_nodes, rankdir, styles)
        print("Network PNG image generation completed")
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz")


def dataset_dummy_input(dataset):
    if dataset == 'imagenet':
        dummy_input = Variable(torch.randn(1, 3, 224, 224), requires_grad=False)
    elif dataset == 'cifar10':
        dummy_input = Variable(torch.randn(1, 3, 32, 32))
    else:
        raise ValueError("Unsupported dataset (%s) - aborting draw operation" % dataset)
    return dummy_input


def export_img_classifier_to_onnx(model, onnx_fname, dataset, export_params=True, add_softmax=True):
    """Export a PyTorch image classifier to ONNX.

    """
    dummy_input = dataset_dummy_input(dataset).to('cuda')
    # Pytorch 0.4 doesn't support exporting modules wrapped in DataParallel
    model = distiller.make_non_parallel_copy(model)

    with torch.onnx.set_training(model, False):
        if add_softmax:
            # Explicitly add a softmax layer, because it is needed for the ONNX inference phase.
            model.original_forward = model.forward
            softmax = torch.nn.Softmax(dim=-1)
            model.forward = lambda input: softmax(model.original_forward(input))
        torch.onnx.export(model, dummy_input, onnx_fname, verbose=False, export_params=export_params)
        msglogger.info('Exported the model to ONNX format at %s' % os.path.realpath(onnx_fname))


def data_node_has_parent(g, id):
    for edge in g.edges:
        if edge.dst == id:
            return True
    return False
