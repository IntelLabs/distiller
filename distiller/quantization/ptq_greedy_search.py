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
from distiller.data_loggers import collect_quant_stats
from distiller.models import create_model
from collections import OrderedDict
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os
import distiller.apputils as apputils

msglogger = None


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
    distiller.modules.BatchMatmul
)

FP16_LAYERS = (
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid
)

PARAM_MODULES = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv3d
)

UNQUANTIZED_MODULES = (
    nn.Softmax,
)

CLIP_MODES = ['NONE',
              'AVG',
              'GAUSS',
              'LAPLACE'
              ]


def module_override(**kwargs):
    override = OrderedDict()
    if kwargs.get('fp16', False):
        override['fp16'] = True
        return override
    return OrderedDict(kwargs)


def ptq_greedy_search(model, dummy_input, eval_fn, calib_eval_fn=None,
                      recurrent=False, classes=CLASSES, act_stats=None):
    """
    Perform greedy search on Post Train Quantization configuration for the model.
    Args:
        model (nn.Module): the model to quantize
        dummy_input (torch.Tensor): a dummy input to be passed to the model
        eval_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        calib_eval_fn (function): An 'evaluation' function to use for forward passing
          through the model to collection quantization calibration statistics.
          if None provided - will use `eval_fn` as a default.
        recurrent (bool): a flag to indicate whether the model has recurrent connections.
        classes (Tuple[type]): a list of types we allow quantization.
        act_stats (OrderedDict): quant calibration activation stats.
            if None provided - will be calculated on runtime.
    Returns:
        (quantized_model, best_overrides_dict)
    Note:
        It is assumed that `eval_fn` returns a satisfying metric of performance (e.g. accuracy)
        and the greedy search aims to maximize this metric.
    """
    best_overrides_dict = OrderedDict()
    overrides_dict = OrderedDict()
    adjacency_map = SummaryGraph(model, dummy_input).adjacency_map()
    modules_to_quantize = layers_quant_order(model, dummy_input, recurrent)
    modules_dict = dict(model.named_modules())
    modules_to_quantize = [m for m in modules_to_quantize if isinstance(modules_dict[m], classes)]
    calib_eval_fn = calib_eval_fn or eval_fn
    if not act_stats:
        print('Collecting stats for model...')
        model_temp = distiller.utils.make_non_parallel_copy(model)
        act_stats = collect_quant_stats(model_temp, calib_eval_fn)
        del model_temp
        print('Done.')
    base_score = eval_fn(model)
    print("Base score: %.3f" % base_score)

    def recalibrate_stats(module_name, act_stats):
        """
        Re-collects quant-calibration stats for successor modules of the current module.
        """
        modules_to_recalibrate = {op.name for op in adjacency_map[module_name].successors} & set(act_stats)
        if not modules_to_recalibrate:
            # either there aren't any successors or
            # the successors aren't in the stats file - skip
            return act_stats
        q = PostTrainLinearQuantizer(distiller.utils.make_non_parallel_copy(model),
                                     bits_activations=None,
                                     bits_parameters=None,
                                     bits_accum=32,
                                     mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                     clip_acts=ClipMode.NONE,
                                     overrides=deepcopy(best_overrides_dict),
                                     model_activation_stats=deepcopy(act_stats))
        q.prepare_model(dummy_input)
        # recalibrate on the current best quantized version of the model.
        recalib_act_stats = collect_quant_stats(q.model, calib_eval_fn, modules_to_collect=modules_to_recalibrate)
        act_stats.update(recalib_act_stats)
        return act_stats

    def fp16_replacement(module, *args):
        return FP16Wrapper(module)

    for module_name in modules_to_quantize:
        print('Searching optimal quantization in \'%s\':' % module_name)
        module = modules_dict[module_name]
        overrides_dict = deepcopy(best_overrides_dict)
        best_performance = float("-inf")
        for clip_mode in CLIP_MODES:
            if isinstance(module, PARAM_MODULES):
                current_module_override = module_override(clip_acts=clip_mode,
                                                          bits_weights=8,
                                                          bits_activations=8,
                                                          bits_bias=32)
            elif isinstance(module, classes):
                current_module_override = module_override(clip_acts=clip_mode,
                                                          bits_weights=8,
                                                          bits_activations=8)
            elif isinstance(module, UNQUANTIZED_MODULES):
                current_module_override = module_override()
            else:
                current_module_override = module_override(fp16=True)

            overrides_dict[module_name] = current_module_override
            if clip_mode == ClipMode.LAPLACE:
                # Since parameter b isn't implemented yet -
                # we don't use activation stats for this module
                # instead we use dynamic quantization:
                temp_act_stats = deepcopy(act_stats)
                temp_act_stats[module_name] = None
            else:
                temp_act_stats = deepcopy(act_stats)
            quantizer = PostTrainLinearQuantizer(deepcopy(model),
                                                 bits_activations=None,
                                                 bits_parameters=None,
                                                 bits_accum=32,
                                                 mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                                 clip_acts=ClipMode.NONE,
                                                 overrides=deepcopy(overrides_dict),
                                                 model_activation_stats=deepcopy(temp_act_stats))
            for fp16_layer_type in FP16_LAYERS:
                quantizer.replacement_factory[fp16_layer_type] = fp16_replacement
            quantizer.prepare_model(dummy_input)

            current_perf = eval_fn(quantizer.model)
            print('\t%s\t score = %.3f\tLayer overrides: %s' %
                  (clip_mode, current_perf, current_module_override))
            if current_perf > best_performance:
                best_overrides_dict[module_name] = current_module_override
                best_performance = current_perf

        # end of search - we update the calibration of the next layers:
        recalibrate_stats(module_name, act_stats)

    quantizer = PostTrainLinearQuantizer(model, mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                         clip_acts=ClipMode.NONE, overrides=deepcopy(best_overrides_dict),
                                         model_activation_stats=act_stats)
    for fp16_layer_type in FP16_LAYERS:
        quantizer.replacement_factory[fp16_layer_type] = fp16_replacement
    quantizer.prepare_model(dummy_input)
    print('best_overrides_dict: %s' % best_overrides_dict)
    print('Best score ', eval_fn(quantizer.model))
    return model, best_overrides_dict


if __name__ == "__main__":
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--qe-calib-portion', type=float, default=1.0,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--qe-calib-batchsize', type=int, default=256,
                        help='The portion of the dataset to use for calibration stats collection.')
    args = parser.parse_args()
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False)

    # quant calibration dataloader:
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, load_train=False, load_val=False)

    msglogger = logging.getLogger()
    logging.disable(logging.WARNING)

    def test_fn(model):
        top1, top5, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        return top1

    def calib_eval_fn(model):
        classifier.test(calib_data_loader, model, cc.criterion, [], None,
                        args)

    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    args.device = next(model.parameters()).device
    if args.load_model_path:
        print("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    dummy_input = torch.rand(*model.input_shape, device=args.device)
    m, overrides = ptq_greedy_search(model, dummy_input, test_fn, calib_eval_fn=calib_eval_fn)
    distiller.yaml_ordered_save('%s.ptq_greedy_search.yaml' % args.arch, overrides)
