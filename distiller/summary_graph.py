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

import distiller
import re
import numpy as np
import collections
import torch
import torch.jit as jit
import logging
from collections import OrderedDict, defaultdict
msglogger = logging.getLogger()


def onnx_name_2_pytorch_name(name):
    # Convert a layer's name from an ONNX name, to a PyTorch name
    # For example:
    #   ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu] ==> layer3.0.relu

    # Split by square brackets
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]

    return '.'.join(name_parts)


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
        self._src_model = model
        model_clone = distiller.make_non_parallel_copy(model)
        with torch.onnx.set_training(model_clone, False):
            
            device = next(model_clone.parameters()).device
            dummy_input = distiller.convert_tensors_recursively_to(dummy_input, device=device)
            trace, _ = jit.get_trace_graph(model_clone, dummy_input, _force_outplace=True)

            # ONNX trace optimization has issues with Gemm ops (aka "Linear" / "addmm" / "FC"), where
            # Gemm nodes get the scope name of the last non-Gemm node that came before them. This can make
            # it impossible, in some cases, to derive the connectivity of the model using the original
            # module names. So we save the scope names for these nodes from the un-optimized trace.
            aten_addmm_nodes_scope_names = [n.scopeName() for n in trace.graph().nodes() if n.kind() == 'aten::addmm']
            onnx_gemm_count = 0

            # Let ONNX do the heavy lifting: fusing the convolution nodes; fusing the nodes
            # composing a GEMM operation; etc.
            torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

            graph = trace.graph()
            self.ops = OrderedDict()
            self.module_ops_map = defaultdict(list)
            self.params = OrderedDict()
            self.edges = []
            self.temp = OrderedDict()

            in_out = list(graph.inputs()) + list(graph.outputs())
            for param in in_out:
                self.__add_param(param)

            for node in graph.nodes():
                new_op = self.__create_op(node)

                # Here we apply the workaround to the Gemm nodes scope name issue mentioned above
                if new_op['type'] == 'Gemm':
                    new_op['orig-name'] = aten_addmm_nodes_scope_names[onnx_gemm_count]
                    new_op['name'] = new_op['orig-name']
                    onnx_gemm_count += 1

                # Convert the graph node's scope name to a PyTorch module name
                module_name = onnx_name_2_pytorch_name(new_op['orig-name'])
                new_op['module-name'] = module_name
                if len(module_name) == 0:
                    # Special case where the module name is an empty string - this happens
                    # when the op is called from the "top-level" of the model
                    new_op['name'] = 'top_level_op'
                else:
                    new_op['name'] = module_name

                # The node's scope name in the graph corresponds to the module from which the op was called.
                # This means that when ops are invoked from the same module via functional calls or direct
                # operations on tensors, these ops will have the SAME MODEL NAME associated with them.
                # For example:
                #   t = t1 + t2
                #   t = F.relu(t)
                # In this case the add operation and the ReLU operation will have the same name, which is
                # derived from the module they're contained in.
                #
                # Another case where different ops will have the same module name is when a module is reused:
                #   out = self.conv1(x)
                #   out = self.relu(out)    <=== First use of self.relu
                #   out = self.conv2(out)
                #   out = self.relu(out)    <=== Second use of self.relu
                # In this case the graph will have 2 distinct ReLU nodes, with the same scope name.
                #
                # Operators with the same name create very confusing graphs (in ResNet, for example),
                # so we "unroll" them.
                same_module_cnt = len(self.module_ops_map[module_name])
                if same_module_cnt:
                    new_op['name'] += "__" + str(same_module_cnt)
                self.module_ops_map[module_name].append(new_op['name'])

                # Finally we register the new op in the ops collection
                msglogger.debug("new sgraph node - Scope name: {} ; Type: {} ; Display name {}".format(
                    new_op['orig-name'], new_op['type'], new_op['name']))
                self.ops[new_op['name']] = new_op

                for input_ in node.inputs():
                    self.__add_input(new_op, input_)
                    self.edges.append(SummaryGraph.Edge(input_.uniqueName(), new_op['name']))

                for output in node.outputs():
                    self.__add_output(new_op, output)
                    self.edges.append(SummaryGraph.Edge(new_op['name'], output.uniqueName()))

                new_op['attrs'] = OrderedDict([(attr_name, node[attr_name]) for attr_name in node.attributeNames()])

        self.__merge_pad_avgpool()
        self.add_macs_attr()
        self.add_footprint_attr()
        self.add_arithmetic_intensity_attr()
        del model_clone

    def __merge_pad_avgpool(self):
        """ The ONNX trace optimization converts average pool ops to a sequence of 2 operations: pad + pool.
        This "quirk" makes makes it unnecessarily difficult to detect the connectivity between an average pool
        op and its predecessor, and it doesn't serve any purpose in the context of SummaryGraph usages.
        So we get rid of the pad op here.
        """
        pad_op_name = None
        for curr_op_name, curr_op in list(self.ops.items()):
            curr_op_type = curr_op['type']
            if curr_op_type == 'Pad':
                pad_op_name = curr_op_name
            else:
                if pad_op_name and curr_op_type == 'AveragePool':
                    pad_op = self.ops[pad_op_name]
                    if pad_op['module-name'] != curr_op['module-name']:
                        continue
                    merged_op = OrderedDict(curr_op)
                    merged_op['name'] = pad_op_name
                    merged_op['inputs'] = pad_op['inputs']
                    self.ops[pad_op_name] = merged_op
                    self.ops.pop(curr_op_name)
                    self.module_ops_map[merged_op['module-name']].remove(curr_op_name)

                    sequence_input_idx = pad_op['inputs'][0]
                    first_edge = SummaryGraph.Edge(sequence_input_idx, pad_op_name)
                    idx = self.edges.index(first_edge)
                    del self.edges[idx:idx + 4]
                    self.edges.insert(idx, SummaryGraph.Edge(sequence_input_idx, pad_op_name))
                    self.edges.insert(idx + 1, SummaryGraph.Edge(pad_op_name, merged_op['outputs'][0]))

                pad_op_name = None

    def __create_op(self, onnx_node):
        op = OrderedDict()
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
        tensor = OrderedDict()
        tensor['id'] = n.uniqueName()
        try:
            # try parsing the FM tensor type.  For example: Float(1, 64, 8, 8)
            s = str(n.node())
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
                groups = op['attrs']['group']
                ofm_vol = self.param_volume(conv_out)
                try:
                    # MACs = volume(OFM) * (#IFM * K^2) / #Groups
                    op['attrs']['MACs'] = int(ofm_vol * SummaryGraph.volume(conv_w) * self.params[conv_in]['shape'][1] / groups)
                except IndexError:
                    # Todo: change the method for calculating MACs
                    msglogger.error("An input to a Convolutional layer is missing shape information "
                                    "(MAC values will be wrong)")
                    msglogger.error("For details see https://github.com/NervanaSystems/distiller/issues/168")
                    op['attrs']['MACs'] = 0
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
                    if op['type'] == 'Conv':
                        kernel_size =  self.volume(op['attrs']['kernel_shape'])
                        group = op['attrs']['group']
                    else:
                        kernel_size, group = 1, 1
                    n_ifm = self.param_shape(conv_in)[1] / group
                    n_ofm = self.param_shape(conv_out)[1] 
                    weights_vol = kernel_size * n_ifm * n_ofm
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
        return self.ops.get(distiller.normalize_module_name(lost_op_name), None)

    def find_param(self, data_name):
        return self.params.get(data_name, None)

    def predecessors(self, node, depth, done_list=None):
        """Returns a list of <op>'s predecessors"""
        if done_list is None:
            done_list = []

        node_name = node['name'] if isinstance(node, dict) else node
        preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                     edge.src not in done_list)]
        done_list += preds

        if depth == 1:
            ret = preds
        else:
            ret = []
            for predecessor in preds:
                ret += self.predecessors(predecessor, depth-1, done_list)

        return [distiller.denormalize_module_name(self._src_model, x) for x in ret]

    def predecessors_f(self, node_name, predecessors_types, done_list=None, logging=None):
        """Returns a list of <op>'s predecessors, if they match the <predecessors_types> criteria.
        """
        node_name = distiller.normalize_module_name(node_name)
        node = self.find_op(node_name)
        node_is_an_op = True
        if node is None:
            node_is_an_op = False
            node = self.find_param(node_name)
            if node is None:
                msglogger.warning("predecessors_f: Could not find node {}".format(node_name))
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
                return [distiller.denormalize_module_name(self._src_model, node_name)]

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

        return [distiller.denormalize_module_name(self._src_model, node) for node in ret]

    def successors(self, node, depth, done_list=None):
        """Returns a list of <op>'s successors"""
        if done_list is None:
            done_list = []

        node_name = node['name'] if isinstance(node, dict) else node
        succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                     edge.dst not in done_list)]
        done_list += succs

        if depth == 1:
            ret = succs
        else:
            ret = []
            for successor in succs:
                ret += self.successors(successor, depth-1, done_list)

        return [distiller.denormalize_module_name(self._src_model, x) for x in ret]

    def successors_f(self, node_name, successors_types, done_list=None, logging=None):
        """Returns a list of <op>'s successors, if they match the <successors_types> criteria.

        Traverse the graph, starting at node <node_name>, and search for successor
        nodes, that have one of the node types listed in <successors_types>.
        If none is found, then return an empty list.

        <node_name> and the returned list of successors are strings, because
        """
        node_name = distiller.normalize_module_name(node_name)
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
                return [distiller.denormalize_module_name(self._src_model, node_name)]

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

        return [distiller.denormalize_module_name(self._src_model, node) for node in ret]

    def named_params_layers(self):
        for param_name, param in self._src_model.named_parameters():
            # remove the extension of param_name, and then normalize it
            # to create a normalized layer name
            normalized_layer_name = distiller.normalize_module_name(
                '.'.join(param_name.split('.')[:-1]))
            sgraph_layer_name = distiller.denormalize_module_name(
                self._src_model, normalized_layer_name)
            yield sgraph_layer_name, param_name, param

    def adjacency_map(self, dedicated_modules_only=False):
        """Returns a mapping from each op in the graph to its immediate predecessors and successors.

        The keys in the generated mapping are op names, and the values are instances of AdjacentsEntry.

        The op names are "de-normalized", meaning they can be used directly with the underlying model's
        named_modules(), for example.

        Args:
            dedicated_modules_only (bool): If set, the generated mapping will not include any ops that can't be
              associated with a dedicated module within the underlying model. Examples of this will be
              functional calls, such as "F.relu()", and tensor operations, such as "t3 = t1 + t2".
        """
        adj_map = OrderedDict()

        for op_name, op in self.ops.items():
            def dedicated_module_check(n):
                module_name = self.ops[distiller.normalize_module_name(n)]['module-name']
                return len(self.module_ops_map[module_name]) == 1 or not dedicated_modules_only

            if not dedicated_module_check(op_name):
                continue

            entry = AdjacentsEntry()
            # Find the immediate preceding and succeeding modules. Depth of 1 gets us the
            # input and output tensors, depth of 2 gets the actual modules
            entry.predecessors = [n for n in self.predecessors(op, 2) if dedicated_module_check(n)]
            entry.successors = [n for n in self.successors(op, 2) if dedicated_module_check(n)]

            adj_map[distiller.denormalize_module_name(self._src_model, op_name)] = entry

        return adj_map


class AdjacentsEntry(object):
    def __init__(self):
        self.predecessors = []
        self.successors = []

    def __repr__(self):
        return 'Predecessors: {0} ; Successors: {1}'.format(self.predecessors, self.successors)
