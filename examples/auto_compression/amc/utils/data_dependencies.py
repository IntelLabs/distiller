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
import distiller

__all__ = ["find_dependencies"]
msglogger = logging.getLogger()


def find_dependencies(dependency_type, sgraph, layers, layer_name, dependencies_list):
    """Produce a list of pruning-dependent layers.

    Given a SummaryGraph instance (sgraph), this function returns the list of dependent
    layers in `dependencies_list`.
    A "dependent layer" is a layer affected by a change in the number of filters/channels
    (dependency_type) in `layer_name`.
    """
    if dependency_type == "channels":
        return _find_dependencies_channels(sgraph, layers, layer_name, dependencies_list)
    if dependency_type == "filters":
        return _find_dependencies_filters(sgraph, layers, layer_name, dependencies_list)
    raise ValueError("%s is not a valid dependency type" % dependency_type)


def _find_dependencies_channels(sgraph, layers, layer_name, dependencies_list):
    # Find all instances of Convolution layers that immediately precede this layer
    predecessors = sgraph.predecessors_f(layer_name, ['Conv'])
    for predecessor in predecessors:
        dependencies_list.append(predecessor)
        prev = sgraph.find_op(predecessor)

        if prev['attrs']['group'] == prev['attrs']['n_ifm']:
            # This is a group-wise convolution, and a special one at that (groups == in_channels).
            _find_dependencies_channels(sgraph, layers, predecessor, dependencies_list)
        elif prev['attrs']['group'] != 1:
            raise ValueError("Distiller AutoCompression currently does not "
                             "handle this convolution groups configuration {} "
                             "(layer={} {}\n{})".format(
                             prev['attrs']['group'], predecessor, prev, layers[predecessor]))


# todo: remove the 'layers' parameter
def _find_dependencies_filters(sgraph, layers, layer_name, dependencies_list):
    # Find all instances of Convolution or FC (GEMM) layers that immediately follow this layer
    successors = sgraph.successors_f(layer_name, ['Conv', 'Gemm'])    
    for successor in successors:
        dependencies_list.append(successor)
        next = sgraph.find_op(successor)
        if next['type'] == 'Conv':
            if next['attrs']['group'] == next['attrs']['n_ifm']:
                # This is a group-wise convolution, and a special one at that (groups == in_channels).
                _find_dependencies_filters(sgraph, layers, successor, dependencies_list)
            elif next['attrs']['group'] != 1:
                raise ValueError("Distiller AutoCompression currently does not handle this conv.groups configuration")    
 