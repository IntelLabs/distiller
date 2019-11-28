#
# Copyright (c) 2019 Intel Corporation
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

import math
import copy
import logging
import torch
import distiller
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name, SummaryGraph


__all__ = ["NetworkWrapper"]
msglogger = logging.getLogger()
ALMOST_ONE = 0.9999

class NetworkWrapper(object):
    def __init__(self, model, app_args, services, modules_list, pruning_pattern):
        self.app_args = app_args
        self.services = services
        self.cached_model_metadata = NetworkMetadata(model, app_args.dataset,
                                                     pruning_pattern, modules_list)
        self.cached_perf_summary = self.cached_model_metadata.performance_summary()
        self.reset(model)
        self.sparsification_masks = None

    def reset(self, model):
        self.model = model
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)
        self.model_metadata = copy.deepcopy(self.cached_model_metadata)

    def get_resources_requirements(self):
        total_macs, total_nnz = self.model_metadata.model_budget()
        return total_macs, total_nnz

    @property
    def arch(self):
        return self.app_args.arch

    def num_pruned_layers(self):
        return self.model_metadata.num_pruned_layers()

    def get_pruned_layer(self, layer_id):
        return self.model_metadata.get_pruned_layer(layer_id)

    def get_layer(self, idx):
       return self.model_metadata.get_layer(idx)

    def layer_macs(self, layer):
        return self.model_metadata.layer_macs(layer)

    def layer_net_macs(self, layer):
        return self.model_metadata.layer_net_macs(layer)

    def name2layer(self, name):
        return self.model_metadata.name2layer(name)

    @property
    def total_macs(self):
        return self.model_metadata.total_macs

    @property
    def total_nnz(self):
        return self.model_metadata.total_nnz

    def performance_summary(self):
        """Return a dictionary representing the performance the model.

        We calculate the performance of each layer relative to the original (uncompressed) model.
        """
        current_perf = self.model_metadata.performance_summary()
        ret = OrderedDict()
        #return OrderedDict({k: v/v_baseline for ((k, v), (v_baseline)) in zip(current_perf.items(), self.cached_perf_summary.values())})
        for k, v in current_perf.items():
            ret[k] = v / self.cached_perf_summary[k]
        return ret

    def create_scheduler(self):
        scheduler = distiller.CompressionScheduler(self.model, self.zeros_mask_dict)
        return scheduler

    def remove_structures(self, layer_id, fraction_to_prune, prune_what, prune_how,
                          group_size, apply_thinning, ranking_noise):
        """Physically remove channels and corresponding filters from the model

        Returns the compute-sparsity of the layer with index 'layer_id'
        """
        if layer_id not in self.model_metadata.pruned_idxs:
            raise ValueError("idx=%d is not in correct range " % layer_id)
        if fraction_to_prune < 0:
            raise ValueError("fraction_to_prune=%.3f is illegal" % fraction_to_prune)
        if fraction_to_prune == 0:
            return 0

        if fraction_to_prune == 1.0:
            # For now, prevent the removal of entire layers
            fraction_to_prune = ALMOST_ONE

        layer = self.model_metadata.get_pruned_layer(layer_id)
        macs_before = self.layer_net_macs(layer)
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.debug("ADC: trying to remove %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            if layer.type == "Linear":
                calculate_sparsity = distiller.sparsity_rows
            remove_structures_fn = distiller.remove_channels
            group_type = "Channels"
        elif prune_what == "filters":
            calculate_sparsity = distiller.sparsity_3D
            group_type = "Filters"
            remove_structures_fn = distiller.remove_filters
        else:
            raise ValueError("unsupported structure {}".format(prune_what))

        if prune_how in ["l1-rank", "stochastic-l1-rank"]:
            # Create a channel/filter-ranking pruner
            pruner = distiller.pruning.L1RankedStructureParameterPruner(
                "auto_pruner", group_type, fraction_to_prune, conv_pname,
                noise=ranking_noise, group_size=group_size)
            meta = None
        elif prune_how == "fm-reconstruction":
            pruner = distiller.pruning.FMReconstructionChannelPruner(
                "auto_pruner", group_type, fraction_to_prune, conv_pname,
                group_size, math.ceil, ranking_noise=ranking_noise)
            meta = {'model': self.model}
        else:
            raise ValueError("Unknown pruning method")
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=meta)
        del pruner

        if (self.zeros_mask_dict[conv_pname].mask is None or
            0 == calculate_sparsity(self.zeros_mask_dict[conv_pname].mask)):
            msglogger.debug("remove_structures: aborting because there are no structures to prune")
            return 0
        final_action = calculate_sparsity(self.zeros_mask_dict[conv_pname].mask)

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        if apply_thinning:
            self.cache_spasification_masks()
            remove_structures_fn(self.model, self.zeros_mask_dict, self.app_args.arch, self.app_args.dataset, optimizer=None)

        self.model_metadata.reduce_layer_macs(layer, final_action)
        macs_after = self.layer_net_macs(layer)
        assert 1. - (macs_after / macs_before) == final_action
        return final_action

    def validate(self):
        top1, top5, vloss = self.services.validate_fn(model=self.model)
        return top1, top5, vloss

    def train(self, num_epochs, episode=0):
        """Train for zero or more epochs"""
        opt_cfg = self.app_args.optimizer_data
        optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_cfg['lr'],
                                    momentum=opt_cfg['momentum'], weight_decay=opt_cfg['weight_decay'])
        compression_scheduler = self.create_scheduler()
        acc_list = []
        for _ in range(num_epochs):
            # Fine-tune the model
            accuracies = self.services.train_fn(model=self.model, compression_scheduler=compression_scheduler,
                                                optimizer=optimizer, epoch=episode)
            acc_list.extend(accuracies)
        del compression_scheduler
        return acc_list

    def cache_spasification_masks(self):
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        self.sparsification_masks = copy.deepcopy(masks)


class NetworkMetadata(object):
    def __init__(self, model, dataset, dependency_type, modules_list):
        details = get_network_details(model, dataset, dependency_type, modules_list)
        self.all_layers, self.pruned_idxs, self.dependent_idxs, self._total_macs, self._total_nnz = details

    @property
    def total_macs(self):
        return self._total_macs

    @property
    def total_nnz(self):
        return self._total_nnz

    def layer_net_macs(self, layer):
        """Returns a MACs of a specific layer"""
        return layer.macs

    def layer_macs(self, layer):
        """Returns a MACs of a specific layer, with the impact on pruning-dependent layers"""
        macs = layer.macs
        for dependent_mod in layer.dependencies:
            macs += self.name2layer(dependent_mod).macs
        return macs

    def reduce_layer_macs(self, layer, reduction):
        total_macs_reduced = layer.macs * reduction
        total_nnz_reduced = layer.weights_vol * reduction
        layer.macs -= total_macs_reduced
        layer.weights_vol -= total_nnz_reduced
        for dependent_mod in layer.dependencies:
            macs_reduced = self.name2layer(dependent_mod).macs * reduction
            nnz_reduced = self.name2layer(dependent_mod).weights_vol * reduction
            total_macs_reduced += macs_reduced
            total_nnz_reduced += nnz_reduced
            self.name2layer(dependent_mod).macs -= macs_reduced
            self.name2layer(dependent_mod).weights_vol -= nnz_reduced
        self._total_macs -= total_macs_reduced
        self._total_nnz -= total_nnz_reduced

    def name2layer(self, name):
        layers = [layer for layer in self.all_layers.values() if layer.name == name]
        if len(layers) == 1:
            return layers[0]
        raise ValueError("illegal module name %s" % name)

    def model_budget(self):
        return self._total_macs, self._total_nnz

    def get_layer(self, layer_id):
        return self.all_layers[layer_id]

    def get_pruned_layer(self, layer_id):
        assert self.is_prunable(layer_id)
        return self.get_layer(layer_id)

    def is_prunable(self, layer_id):
        return layer_id in self.pruned_idxs

    def is_compressible(self, layer_id):
        return layer_id in (self.pruned_idxs + self.dependent_idxs)

    def num_pruned_layers(self):
        return len(self.pruned_idxs)

    def num_layers(self):
        return len(self.all_layers)

    def performance_summary(self):
        # return OrderedDict({layer.name: (layer.macs, layer.weights_vol)
        #                    for layer in self.all_layers.values()})
        return OrderedDict({layer.name: layer.macs
                           for layer in self.all_layers.values()})
    

def get_network_details(model, dataset, dependency_type, layers_to_prune=None):
    def make_conv(model, conv_module, g, name, seq_id, layer_id):
        conv = SimpleNamespace()
        conv.type = "Conv2D"
        conv.name = name
        conv.id = layer_id
        conv.t = seq_id
        conv.k = conv_module.kernel_size[0]
        conv.stride = conv_module.stride

        # Use the SummaryGraph to obtain some other details of the models
        conv_op = g.find_op(normalize_module_name(name))
        assert conv_op is not None

        conv.weights_vol = conv_op['attrs']['weights_vol']
        conv.macs = conv_op['attrs']['MACs']
        conv.n_ofm = conv_op['attrs']['n_ofm']
        conv.n_ifm = conv_op['attrs']['n_ifm']
        conv_pname = name + ".weight"
        conv_p = distiller.model_find_param(model, conv_pname)
        conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
        conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
        conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
        conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]
        return conv

    def make_fc(model, fc_module, g, name, seq_id, layer_id):
        fc = SimpleNamespace()
        fc.type = "Linear"
        fc.name = name
        fc.id = layer_id
        fc.t = seq_id

        # Use the SummaryGraph to obtain some other details of the models
        fc_op = g.find_op(normalize_module_name(name))
        assert fc_op is not None

        fc.weights_vol = fc_op['attrs']['weights_vol']
        fc.macs = fc_op['attrs']['MACs']
        fc.n_ofm = fc_op['attrs']['n_ofm']
        fc.n_ifm = fc_op['attrs']['n_ifm']
        fc_pname = name + ".weight"
        fc_p = distiller.model_find_param(model, fc_pname)
        fc.ofm_h = g.param_shape(fc_op['outputs'][0])[0]
        fc.ofm_w = g.param_shape(fc_op['outputs'][0])[1]
        fc.ifm_h = g.param_shape(fc_op['inputs'][0])[0]
        fc.ifm_w = g.param_shape(fc_op['inputs'][0])[1]

        return fc

    dummy_input = distiller.get_dummy_input(dataset)
    g = SummaryGraph(model, dummy_input)
    all_layers = OrderedDict()
    pruned_indices = list()
    dependent_layers = set()
    total_macs = 0
    total_params = 0

    unfiltered_layers = layers_topological_order(model, dummy_input)
    mods = dict(model.named_modules())
    layers = OrderedDict({mod_name: mods[mod_name] for mod_name in unfiltered_layers
                          if mod_name in mods and
                          isinstance(mods[mod_name], (torch.nn.Conv2d, torch.nn.Linear))})

    # layers = OrderedDict({mod_name: m for mod_name, m in model.named_modules()
    #                       if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))})
    for layer_id, (name, m) in enumerate(layers.items()):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Conv2d):
                new_layer = make_conv(model, m, g, name, seq_id=len(pruned_indices), layer_id=layer_id)
                all_layers[layer_id] = new_layer
                total_params += new_layer.weights_vol
                total_macs += new_layer.macs
            elif isinstance(m, torch.nn.Linear):
                new_layer = make_fc(model, m, g, name, seq_id=len(pruned_indices), layer_id=layer_id)
                all_layers[layer_id] = new_layer
                total_params += new_layer.weights_vol
                total_macs += new_layer.macs

            if layers_to_prune is None or name in layers_to_prune:
                pruned_indices.append(layer_id)
                # Find the data-dependent layers of this convolution
                from utils.data_dependencies import find_dependencies
                new_layer.dependencies = list()
                find_dependencies(dependency_type, g, all_layers, name, new_layer.dependencies)
                dependent_layers.add(tuple(new_layer.dependencies))

    def convert_layer_names_to_indices(layer_names):
        """Args:
            layer_names - list of layer names
           Returns:
            list of layer indices
        """
        layer_indices = [index for name in layer_names for index,
                            layer in all_layers.items() if layer.name == name[0]]
        return layer_indices

    dependent_indices = convert_layer_names_to_indices(dependent_layers)
    return all_layers, pruned_indices, dependent_indices, total_macs, total_params


def layers_topological_order(model, dummy_input, recurrent=False):
    """
    Prepares an ordered list of layers to quantize sequentially. This list has all the layers ordered by their
    topological order in the graph.
    Args:
        model (nn.Module): the model to quantize.
        dummy_input (torch.Tensor): an input to be passed through the model.
        recurrent (bool): indication on whether the model might have recurrent connections.
    """

    class _OpRank:
        def __init__(self, adj_entry, rank=None):
            self.adj_entry = adj_entry
            self._rank = rank or 0

        @property
        def rank(self):
            return self._rank

        @rank.setter
        def rank(self, val):
            self._rank = max(val, self._rank)

        def __repr__(self):
            return '_OpRank(\'%s\' | %d)' % (self.adj_entry.op_meta.name, self.rank)

    adj_map = SummaryGraph(model, dummy_input).adjacency_map()
    ranked_ops = {k: _OpRank(v, 0) for k, v in adj_map.items()}

    def _recurrent_ancestor(ranked_ops_dict, dest_op_name, src_op_name):
        def _is_descendant(parent_op_name, dest_op_name):
            successors_names = [op.name for op in adj_map[parent_op_name].successors]
            if dest_op_name in successors_names:
                return True
            for succ_name in successors_names:
                if _is_descendant(succ_name, dest_op_name):
                    return True
            return False

        return _is_descendant(dest_op_name, src_op_name) and \
            (0 < ranked_ops_dict[dest_op_name].rank < ranked_ops_dict[src_op_name].rank)

    def rank_op(ranked_ops_dict, op_name, rank):
        ranked_ops_dict[op_name].rank = rank
        for child_op in adj_map[op_name].successors:
            # In recurrent models: if a successor is also an ancestor - we don't increment its rank.
            if not recurrent or not _recurrent_ancestor(ranked_ops_dict, child_op.name, op_name):
                rank_op(ranked_ops_dict, child_op.name, ranked_ops_dict[op_name].rank + 1)

    roots = [k for k, v in adj_map.items() if len(v.predecessors) == 0]
    for root_op_name in roots:
        rank_op(ranked_ops, root_op_name, 0)

     # Take only the modules from the original model
    # module_dict = dict(model.named_modules())
    # Neta
    ret = sorted([k for k in ranked_ops.keys()],
                 key=lambda k: ranked_ops[k].rank)

    # Check that only the actual roots have a rank of 0
    assert {k for k in ret if ranked_ops[k].rank == 0} <= set(roots)
    return ret
