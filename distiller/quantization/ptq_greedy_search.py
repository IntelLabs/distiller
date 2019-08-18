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
r"""
Here we implement the greedy search algorithm for automatic quantization.
"""
import torch
import torch.nn as nn
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, LinearQuantMode, FP16Wrapper
from distiller.summary_graph import SummaryGraph
import distiller.modules
import distiller.data_loggers
from distiller.models import create_model
from collections import OrderedDict
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os

msglogger = logging.getLogger()


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


def layers_quant_order(model, dummy_input, recurrent=False):
    """
    Prepares an ordered list of layers to quantize sequentially. This list has all the layers ordered by their
    topological order in the graph.
    Args:
        model (nn.Module): the model to quantize.
        dummy_input (torch.Tensor): an input to be passed through the model.
        recurrent (bool): indication on whether the model might have recurrent connections.
    """
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
    module_dict = dict(model.named_modules())
    ret = sorted([k for k in ranked_ops.keys() if k in module_dict],
                 key=lambda k: ranked_ops[k].rank)
    # Check that only the actual roots have a rank of 0
    assert {k for k in ret if ranked_ops[k].rank == 0} <= set(roots)
    return ret


CLASSES = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv3d,
    distiller.modules.Concat,
    distiller.modules.EltwiseAdd,
    distiller.modules.EltwiseMult,
    distiller.modules.Matmul,
    distiller.modules.BatchMatmul,
    nn.Embedding
)

FP16_LAYERS = (
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid
)


def ptq_greedy_search(model, dummy_input, eval_fn, recurrent=False, classes=CLASSES, act_stats=None):
    """
    Perform greedy search on Post Train Quantization configuration for the model.
    Args:
        model (nn.Module): the model to quantize
        dummy_input (torch.Tensor): a dummy input to be passed to the model
        eval_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        recurrent (bool): a flag to indicate whether the model has recurrent connections.
        classes (Tuple[type]): a list of types we allow quantization.
        act_stats (OrderedDict): quant calibration activation stats.
            if None provided - will be calculated on runtime.
    Returns:
        (quantized_model, best_overrides_dict)
    Note:
        It is assumed that `eval_fn` returns a satisfying metric of performance (e.g. accuracy)
        and the greedy search aims to maximize this metric
    """
    clip_modes_to_search = [ClipMode.NONE, ClipMode.AVG, ClipMode.GAUSS, ClipMode.LAPLACE]
    best_overrides_dict = OrderedDict()
    overrides_dict = OrderedDict()
    modules_to_quantize = layers_quant_order(model, dummy_input, recurrent)
    modules_dict = dict(model.named_modules())
    modules_to_quantize = [m for m in modules_to_quantize if isinstance(modules_dict[m], classes)]
    if not act_stats:
        print('Collecting stats for model...')
        model_temp = deepcopy(model)
        act_stats = distiller.data_loggers.collect_quant_stats(model_temp, eval_fn)
        del model_temp
        print('Done.')
    best_act_stats = None
    base_score = eval_fn(model)
    print("Base score: %.3f" % base_score)
    def fp16_replacement(module, *args):
        return FP16Wrapper(module)
    first_printed = False
    for module_name in modules_to_quantize:
        print('Searching optimal quantization in \'%s\':' % module_name)
        current_module_override = OrderedDict()
        current_module_override['clip_acts'] = ClipMode.NONE
        current_module_override['bits_parameters'] = 8
        current_module_override['bits_activations'] = 8
        current_module_override['bits_accum'] = 32
        overrides_dict[module_name] = current_module_override
        best_performance = float("-inf")
        for clip_mode in clip_modes_to_search:
            if clip_mode == ClipMode.LAPLACE:
                # Since parameter b isn't implemented yet -
                # we don't use activation stats for this module
                # instead we use dynamic quantization:
                temp_act_stats = deepcopy(act_stats)
                temp_act_stats[module_name] = OrderedDict()
            else:
                temp_act_stats = deepcopy(act_stats)
            current_module_override['clip_acts'] = clip_mode
            quantizer = PostTrainLinearQuantizer(deepcopy(model),
                                                 mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                                 clip_acts=ClipMode.NONE,
                                                 overrides=overrides_dict,
                                                 model_activation_stats=deepcopy(temp_act_stats))
            for fp16_layer_type in FP16_LAYERS:
                quantizer.replacement_factory[fp16_layer_type] = fp16_replacement
            quantizer.prepare_model(dummy_input)
            if not first_printed:
                print(quantizer.model)
                first_printed = True
            current_perf = eval_fn(quantizer.model)
            print('\tLayer: %s\t%s\t score = %.3f' % (module_name, clip_mode, current_perf))
            if current_perf > best_performance:
                best_overrides_dict[module_name] = current_module_override
                best_act_stats = temp_act_stats

    quantizer = PostTrainLinearQuantizer(model, mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                         clip_acts=ClipMode.NONE, overrides=best_overrides_dict,
                                         model_activation_stats=act_stats)
    for fp16_layer_type in FP16_LAYERS:
        quantizer.replacement_factory[fp16_layer_type] = fp16_replacement
    quantizer.prepare_model(dummy_input)
    print('best_overrides_dict: %s' % best_overrides_dict)
    return model, best_overrides_dict


if __name__ == "__main__":
    args = classifier.init_classifier_compression_arg_parser().parse_args()
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    data_loader = classifier.load_data(args, load_train=False, load_val=False)

    def test_fn(model):
        top1, top5, losses = classifier.test(data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        return top1

    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    dummy_input = torch.rand(*model.input_shape, device=next(model.parameters()).device)
    ptq_greedy_search(model, dummy_input, test_fn)
