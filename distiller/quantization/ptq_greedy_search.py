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
from distiller.model_transforms import fold_batch_norms
import distiller.modules
from distiller.data_loggers import collect_quant_stats
from distiller.models import create_model
from collections import OrderedDict
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os
import distiller.apputils as apputils
import re

__all__ = ['ptq_greedy_search']

msglogger = None

QUANTIZED_MODULES = (
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

SKIP_MODULES = (
    nn.Identity,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d
)

CLIP_MODES = ['NONE',
              'AVG',
              'GAUSS',
              'LAPLACE'
              ]


def override_odict(**kwargs):
    return OrderedDict(kwargs)


def get_input_modules(sg, recurrent=False):
    """
    Finds the modules in the graph that take the user input directly
    Args:
        sg (SummaryGraph): the summary graph of the model
        recurrent: see SummaryGraph.layers_topological_order
    """
    input_modules = set()
    layers = set(sg.layers_topological_order(recurrent))
    for op in sg.top_level_ops():
        input_modules.update(set(sg.successors(op, 1)) & layers)
    return list(input_modules)


def input_override_generator(module, module_name, sg, overrides_dict, **kwargs):
    """
    Generator for overrides on inputs of the input layers.
    Args:
        module (nn.Module): the module
        module_name (str): module name as it appears in the summary graph
        sg (SummaryGraph): a summary graph of the model
        overrides_dict (OrderedDict): the fixed overrides already applied
        kwargs: additional arguments, if needed
    """
    input_nodes = sg.predecessors(module_name, 1)
    input_idx = kwargs.get('input_idx', 0)
    assert input_idx < len(input_nodes)
    for clip_mode in CLIP_MODES:
        input_idx_override = override_odict(bits_activations=8,
                                            clip_acts=clip_mode)
        input_overrides = OrderedDict([(input_idx, input_idx_override)])
        current_module_override = override_odict(input_overrides=input_overrides)
        yield current_module_override


def module_override_generator(module, module_name, sg, overrides_dict, **kwargs):
    """
    Standard generator of overrides for the greedy search algorithm.
    Args:
        module (nn.Module): the module
        module_name (str): module name as it appears in the summary graph
        sg (SummaryGraph): a summary graph of the model
        overrides_dict (OrderedDict): the fixed overrides already applied
        kwargs: additional arguments, if needed
    """
    adj_map = sg.adjacency_map()
    layers_quant_order = sg.layers_topological_order()
    modules_dict = dict(sg._src_model.named_modules())
    # Quantize input explicitly:
    quantize_input = len(adj_map[module_name].predecessors) == 0
    successors_names = {op.name for op in adj_map[module_name].successors if op.name in modules_dict}
    fake = len(set(layers_quant_order) & set(successors_names)) > 0
    use_half_range = all([isinstance(modules_dict[succ], nn.ReLU) for succ in successors_names])
    use_fake = False
    fpq_module = None
    if isinstance(module, FP16_LAYERS):
        fpq_module = 16
        use_fake = fake
    if isinstance(module, UNQUANTIZED_MODULES) or not isinstance(module, QUANTIZED_MODULES):
        fpq_module = 32
        use_fake = fake
    for clip_mode in CLIP_MODES:
        if isinstance(module, PARAM_MODULES):
            current_module_override = override_odict(clip_acts=clip_mode,
                                                     bits_weights=8,
                                                     bits_activations=8,
                                                     bits_bias=32)
        else:
            current_module_override = override_odict(clip_acts=clip_mode,
                                                     fpq_module=fpq_module,
                                                     fake=use_fake,
                                                     bits_weights=8,
                                                     bits_activations=8)
        if quantize_input:
            current_module_override['input_overrides'] = OrderedDict([(0, override_odict(bits_activations=8))])
        current_module_override['clip_half_range'] = use_half_range and clip_mode in ['GAUSS', 'LAPLACE']

        yield current_module_override


def search_best_local_settings(module, module_name, sg, eval_fn, best_overrides_dict, override_gen_fn, **kwargs):
    msglogger.info('Searching optimal quantization in \'%s\':' % module_name)
    overrides_dict = deepcopy(best_overrides_dict)
    best_performance, best_local_override = float("-inf"), OrderedDict()
    normalized_module_name = module_name
    if isinstance(model, nn.DataParallel):
        normalized_module_name = re.sub(r'module\.', '', normalized_module_name)
    for local_override in override_gen_fn(module, module_name, sg, best_overrides_dict, **kwargs):
        overrides_dict[normalized_module_name] = local_override
        temp_act_stats = deepcopy(act_stats)
        quantizer = PostTrainLinearQuantizer(deepcopy(model),
                                             bits_activations=None,
                                             bits_parameters=None,
                                             bits_accum=32,
                                             mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                             clip_acts=ClipMode.NONE,
                                             overrides=deepcopy(overrides_dict),
                                             model_activation_stats=deepcopy(temp_act_stats),
                                             inputs_quant_auto_fallback=True)
        quantizer.prepare_model(dummy_input)

        current_performance = eval_fn(quantizer.model)
        if not isinstance(module, UNQUANTIZED_MODULES):
            clip_mode = local_override['clip_acts']
            msglogger.info('\t%s\t score = %.3f\tLayer overrides: %s' %
                           (clip_mode, current_performance, local_override))
        else:
            msglogger.info('\t Module is not quantized to int8. Not clipping activations.')
            msglogger.info('\t score = %.3f\tLayer overrides: %s' %
                           (current_performance, local_override))
        if current_performance > best_performance:
            best_performance = current_performance
            best_local_override = local_override

    msglogger.info('\t Choosing overrides: %s' % best_local_override)
    return best_local_override


def ptq_greedy_search(model, dummy_input, eval_fn, calib_eval_fn=None,
                      recurrent=False, act_stats=None,
                      args=None,
                      module_override_gen_fn=None, input_override_gen_fn=None,
                      fold_sequences=True):
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
        act_stats (OrderedDict): quant calibration activation stats.
          if None provided - will be calculated on runtime.
        args: command line arguments
        module_override_gen_fn: A function to generate module overrides.
          assumes signature
          `def module_override_gen_fn(module: nn.Module,
                                      module_name: str,
                                      sg: distiller.SummaryGraph,
                                      overrides_dict: OrderedDict,
                                      **kwargs)-> Generator[OrderedDict, None, None]`
        input_override_gen_fn: Same as module_override_gen_fn, only quantized inputs to the top level layers.
        fold_sequences (bool): fold batch norms before quantizing
    Returns:
        (quantized_model, best_overrides_dict)
    Note:
        It is assumed that `eval_fn` returns a satisfying metric of performance (e.g. accuracy)
        and the greedy search aims to maximize this metric.
    """
    if fold_sequences:
        model = fold_batch_norms(model, dummy_input)
    best_overrides_dict = OrderedDict()
    overrides_dict = OrderedDict()
    sg = SummaryGraph(model, dummy_input)
    modules_to_quantize = sg.layers_topological_order(recurrent)
    adjacency_map = sg.adjacency_map()
    modules_dict = dict(model.named_modules())
    modules_to_quantize = [m for m in modules_to_quantize
                           if m not in args.qe_no_quant_layers]

    module_override_gen_fn = module_override_gen_fn or module_override_generator
    input_override_gen_fn = input_override_gen_fn or input_override_generator

    calib_eval_fn = calib_eval_fn or eval_fn
    if not act_stats:
        msglogger.info('Collecting stats for model...')
        model_temp = distiller.utils.make_non_parallel_copy(model)
        act_stats = collect_quant_stats(model_temp, calib_eval_fn)
        del model_temp
        if args:
            act_stats_path = '%s_act_stats.yaml' % args.arch
            msglogger.info('Done. Saving act stats into %s' % act_stats_path)
            distiller.yaml_ordered_save(act_stats_path, act_stats)
    msglogger.info('Evaluating baseline score for model...')
    base_score = args.base_score or eval_fn(model)
    msglogger.info("Base score: %.3f" % base_score)

    def recalibrate_stats(module_name, act_stats):
        """
        Re-collects quant-calibration stats for successor modules of the current module.
        """
        msglogger.info('Recalibrating stats...')
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
                                     model_activation_stats=deepcopy(act_stats),
                                     inputs_quant_auto_fallback=True)
        q.prepare_model(dummy_input)
        # recalibrate on the current best quantized version of the model.
        recalib_act_stats = collect_quant_stats(q.model, calib_eval_fn, modules_to_collect=modules_to_recalibrate)
        msglogger.info('Done.')
        act_stats.update(recalib_act_stats)
        return act_stats
    # Quantize inputs:
    input_modules = get_input_modules(sg, recurrent)  # top level modules
    for module_name in input_modules:
        module = modules_dict[module_name]
        if isinstance(module, SKIP_MODULES):
            msglogger.info('Skipping module \'%s\' of type %s.' % (module_name, type(module)))
            continue
        msglogger.info('Quantizing top level inputs for %s' % module_name)

        normalized_module_name = module_name
        if isinstance(model, nn.DataParallel):
            normalized_module_name = re.sub(r'module\.', '', normalized_module_name)
        if not best_overrides_dict.get(normalized_module_name, None):
            best_overrides_dict[normalized_module_name] = OrderedDict()
        input_idxs = [i for i, op in enumerate(sg.predecessors(module_name, 1)) if op in sg.top_level_ops()]
        for input_idx in input_idxs:
            best_module_override = search_best_local_settings(module, module_name, sg, eval_fn, best_overrides_dict,
                                                              input_override_gen_fn, input_idx=input_idx)
            best_overrides_dict[normalized_module_name].update(best_module_override)

    # Quantize layers as a whole:
    for module_name in modules_to_quantize:
        module = modules_dict[module_name]
        if isinstance(module, SKIP_MODULES):
            msglogger.info('Skipping module \'%s\' of type %s.' % (module_name, type(module)))
            continue

        normalized_module_name = module_name
        if isinstance(model, nn.DataParallel):
            normalized_module_name = re.sub(r'module\.', '', normalized_module_name)
        if not best_overrides_dict.get(normalized_module_name, None):
            best_overrides_dict[normalized_module_name] = OrderedDict()

        best_module_override = search_best_local_settings(module, module_name, sg, eval_fn, best_overrides_dict,
                                                          module_override_gen_fn)
        best_overrides_dict[normalized_module_name].update(best_module_override)

        # end of search - we update the calibration of the next layers:
        recalibrate_stats(module_name, act_stats)

    quantizer = PostTrainLinearQuantizer(model, mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                         clip_acts=ClipMode.NONE, overrides=deepcopy(best_overrides_dict),
                                         model_activation_stats=act_stats)
    quantizer.prepare_model(dummy_input)
    msglogger.info('best_overrides_dict: %s' % best_overrides_dict)
    msglogger.info('Best score ', eval_fn(quantizer.model))
    return model, best_overrides_dict


def config_verbose(verbose):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
        logging.getLogger().setLevel(logging.WARNING)
    for module in ["distiller.apputils.image_classifier", ]:
        logging.getLogger(module).setLevel(loglevel)


if __name__ == "__main__":
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--qe-no-quant-layers', '--qenql', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                       help='List of layer names for which to skip quantization.')
    parser.add_argument('--qe-calib-portion', type=float, default=1.0,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--qe-calib-batchsize', type=int, default=256,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--base-score', type=float, default=None)
    args = parser.parse_args()
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False)

    # quant calibration dataloader:
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, load_train=False, load_val=False)
    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)

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
        msglogger.info("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    dummy_input = torch.rand(*model.input_shape, device=args.device)
    if args.qe_stats_file:
        msglogger.info("Loading stats from %s" % args.qe_stats_file)
        with open(args.qe_stats_file, 'r') as f:
            act_stats = distiller.yaml_ordered_load(f)
    else:
        act_stats = None
    m, overrides = ptq_greedy_search(model, dummy_input, test_fn,
                                     calib_eval_fn=calib_eval_fn, args=args,
                                     act_stats=act_stats)
    distiller.yaml_ordered_save('%s.ptq_greedy_search.yaml' % args.arch, overrides)