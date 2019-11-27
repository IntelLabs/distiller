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
import torch
import torch.nn as nn
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, \
    RangeLinearQuantWrapper, RangeLinearEmbeddingWrapper, is_post_train_quant_wrapper
from functools import partial
from distiller.summary_graph import SummaryGraph
from distiller.model_transforms import fold_batch_norms
import distiller.modules
from distiller.data_loggers import collect_quant_stats
from distiller.models import create_model
from collections import OrderedDict
from itertools import count
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os
import distiller.apputils as apputils
import re
import argparse
import scipy.optimize as opt
import numpy as np


def quant_params_dict2vec(p_dict):
    keys, vals = zip(*p_dict.items())  # unzip the list of tuples
    vals = np.array([val.item() for val in vals])
    return keys, vals


def quant_params_vec2dict(keys, vals):
    return OrderedDict(zip(keys, (val.item() for val in vals)))


def lp_loss(x, y, p):
    return (torch.sum(torch.abs(x-y)**p)**(1/p)).item()


l1_loss = partial(lp_loss, p=1)
l2_loss = partial(lp_loss, p=2)
l3_loss = partial(lp_loss, p=3)


_INIT_MODES = {
    'NONE': ClipMode.NONE, 'AVG': ClipMode.AVG, 'LAPLACE': ClipMode.LAPLACE, 'GAUSS': ClipMode.GAUSS,
    'L1': l1_loss, 'L2': l2_loss, 'L3': l3_loss
}


def _init_mode_from_str(init_mode_str):
    if init_mode_str not in _INIT_MODES:
        raise ValueError('Unsupported init mode \'%s\'. '
                         'The supported init modes are: %s.' % (init_mode_str, _INIT_MODES))
    return _INIT_MODES[init_mode_str]


def optimize_for_layer(layer, quantized_layer, loss_fn, input, method=None, search_on=[]):
    """
    Searches for optimal linear quantization parameters (scale, zero_point) for a layer
    with respect to the loss function. Assumes loss_fn is of the signature `loss_fn(y, y_q)->float`
    Args:
        layer (nn.Module): the original, pre-quantized, layer.
        quantized_layer (RangeLinearQuantWrapper or RangeLinearEmbeddingWrapper): the post-quantized layer.
        loss_fn (callable): the loss function to optimize with respect to it.
        method (str or callable): the method of optimization, as will be used by scipy.optimize.minimize.
        search_on (list): list of quant_param names to search on. Choices : 'scale' and 'zero_point'.
    Returns:
        quantized_layer after optimization
    """
    init_qp_dict = OrderedDict(quantized_layer.named_linear_quant_params())
    init_qp_dict = OrderedDict((k, v) for k, v in init_qp_dict.items() if
                               any(b in k for b in search_on))

    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict)

    def feed_forward_fn(qp_vec):
        qp_dict = quant_params_vec2dict(keys, qp_vec)
        quantized_layer.update_linear_quant_params(qp_dict)
        y, q_y = layer(*input), quantized_layer(*input)
        return loss_fn(y, q_y)

    _iter = count(1)

    result = opt.minimize(feed_forward_fn, init_qp_vec, method=method)  # type: opt.OptimizeResult
    qp_dict = quant_params_vec2dict(keys, result.x)
    quantized_layer.update_linear_quant_params(qp_dict)
    return quantized_layer


def get_input_for_layer(model, layer_name, input_for_model):
    layer = dict(model.named_modules())[layer_name]  # type: nn.Module
    layer_inputs = []

    def hook_layer_input(module, input, output):
        layer_inputs.append(input)

    handle = layer.register_forward_hook(hook_layer_input)
    output = model(input_for_model)
    assert len(layer_inputs) == 1
    handle.remove()
    return layer_inputs[0]


def init_layer_linear_quant_params(quantizer, original_model, layer_name, init_mode,
                                   init_mode_method='Powell', sample_input=None, search_on=[]):
    """
    Initializes a layer's linear quant parameters.
    This is done to set the scipy.optimize.minimize initial guess.
    Args:
        quantizer (PostTrainLinearQuantizer): the quantizer, **after** calling prepare model.
        original_model (nn.Module): the original, pre-quantized, model.
        layer_name (str): the name of the layer.
        init_mode (ClipMode or callable or str): the initialization mode.
          If ClipMode, the initialization will be according to the respective ClipMode.
          If callable - init_mode will be treated as a loss function between the activations pre and post-quantization,
            and the initialization process will attempt to find the minimum of that loss function.
            E.g. if l1_loss has been passed, the initialization vector will be
              scale, zero_point = argmin_{s, zp} (l1_loss(layer(input), q_layer(input; s, zp)))
          If str - the mode will be chosen from a list of options. The options are:
            [NONE, AVG, LAPLACE, GAUSS, L1, L2 ,L3].
          Defaults to ClipMode.NONE
        init_mode_method (str or callable): applicable only in the case of init_mode = 'L1/2/3' or callable.
          chooses the minimization method for finding the local argmin_{s, zp}.
          Defaults to 'Powell'
        sample_input : a sample batch input for the model.
          Not to be confused with dummy_input - this input has to be an actual batch from a data loader.
          applicable only in the case of init_mode = 'L1/2/3' or callable.
        search_on (list): list of quant_param names to search on. Choices : 'scale' and 'zero_point'.
    """
    if isinstance(init_mode, str):
        init_mode = _init_mode_from_str(init_mode)
    if isinstance(init_mode, ClipMode):
        quantizer.module_overrides_map[layer_name]['clip_acts'] = init_mode
    layer = dict(original_model.named_modules())[layer_name]
    local_args, local_kwargs = quantizer.modules_processed_args[layer_name]
    replace_fn = quantizer.replacement_factory.get(type(layer), quantizer.default_repalcement_fn)
    quantized_layer = replace_fn(deepcopy(layer), *local_args, **local_kwargs)
    if not is_post_train_quant_wrapper(quantized_layer, False):
        return quantized_layer  # the module wasn't quantized, nothing to do here

    if callable(init_mode):
        input_for_layer = get_input_for_layer(original_model, layer_name, sample_input)
        quantized_layer = optimize_for_layer(layer, quantized_layer, init_mode, input_for_layer, init_mode_method,
                                             search_on=search_on)

    distiller.model_setattr(quantizer.model, layer_name, quantized_layer)


def init_linear_quant_params(quantizer, original_model, sample_input, init_mode,
                             init_mode_method=None, search_on=[]):
    """
    Initializes all linear quantization parameters of the model.
    Args:
        quantizer (PostTrainLinearQuantizer): the quantizer, **after** calling prepare model.
        original_model (nn.Module): the original, pre-quantized, model.
        init_mode (ClipMode or callable or str or dict): See `init_layer_linear_qaunt_params`.
          if init_mode is dict - init_mode is configuration for the different layers,
          i.e. init_mode = Dict[layer_name:str, init_mode_layer: ClipMode or callable or str].
        sample_input : a sample batch input for the model.
          Not to be confused with dummy_input - this input has to be an actual batch from a data loader.
          Note - unlike in `init_layer_linear_quant_params`, this argument is required here.
        init_mode_method: See `init_layer_linear_qaunt_params`.
        search_on (list): list of quant_param names to search on. Choices : 'scale' and 'zero_point'.
    """
    layers_topological_order = SummaryGraph(original_model, sample_input).layers_topological_order()
    for module_name in layers_topological_order:
        # check to see if it was quantized:
        q_module = dict(quantizer.model.named_modules())[module_name]
        if not is_post_train_quant_wrapper(q_module, False):
            continue
        module_init_mode = init_mode[module_name] if isinstance(init_mode, dict) else init_mode
        msglogger.debug('Initializing layer \'%s\' using %s mode' % (module_name, module_init_mode))
        init_layer_linear_quant_params(quantizer, original_model, module_name, module_init_mode,
                                       init_mode_method=init_mode_method,
                                       sample_input=sample_input,
                                       search_on=search_on)
    quantizer._post_prepare_model()


def get_default_args():
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--qe-calib-portion', type=float, default=1.0,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--qe-calib-batchsize', type=int, default=256,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--opt-maxiter', dest='maxiter', default=None, type=int,
                        help='Max iteration for minimization method.')
    parser.add_argument('--opt-maxfev', dest='maxfev', default=None, type=int,
                        help='Max iteration for minimization method.')
    parser.add_argument('--opt-method', dest='method', default='Powell',
                        help='Minimization method used by scip.optimize.minimize.')
    parser.add_argument('--opt-bh', dest='basinhopping', action='store_true', default=False,
                        help='Use scipy.optimize.basinhopping stochastic global minimum search.')
    parser.add_argument('--opt-bh-niter', dest='niter', default=100,
                        help='Number of iterations for the basinhopping algorithm.')
    parser.add_argument('--opt-init-mode', dest='init_mode', default='NONE',
                        choices=list(_INIT_MODES),
                        help='The mode of quant initalization. Choices: ' + '|'.join(list(_INIT_MODES)))
    parser.add_argument('--opt-init-method', dest='init_mode_method',
                        help='If --opt-init-mode was specified as L1/L2/L3, this specifies the method of '
                             'minimization.')
    parser.add_argument('--opt-test-size', type=float, default=1,
                        help='Use portion of the test size.')
    parser.add_argument('--search-for-weights', dest='save_fp_weights', action='store_true', default=False,
                        help='Whether or not search quantization parameters for weights as well.')
    parser.add_argument('--search-on', nargs='+', type=str, default=[],
                        help='Which buffers to conduct the search on. Choices: \'scale\' and \'zero_point\'. '
                             'Default: both.')
    parser.add_argument('--base-score', type=float, default=None)
    args = parser.parse_args()
    return args


def ptq_coordinate_search(model, sample_input, eval_fn, method='Powell', options=None, calib_eval_fn=None,
                          act_stats=None, args=None, fold_sequences=True, basinhopping=False,
                          init_args=None, minimizer_kwargs=None,
                          test_fn=None):
    """
    Searches for the optimal post-train quantization configuration (scale/zero_points)
    for a model using numerical methods, as described by scipy.optimize.minimize.
    Args:
        model (nn.Module): model to quantize
        sample_input: an sample expected input to the model
        eval_fn (callable): evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
        method (str or callable): minimization method as accepted by scipy.optimize.minimize.
        options (dict or None): options for the scipy optimizer
        calib_eval_fn (callable): quant-stats calibration evaluation function.
        act_stats (OrderedDict): dictionary of statistics per layer, including inputs and outputs.
          for more context refer to collect_quant_stats.
        args: arguments from command-line.
        fold_sequences (bool): flag, indicates to fold sequences before performing the search.
        basinhopping (bool): flag, indicates to use basinhopping as a global-minimization method,
          will pass the `method` argument to `scipy.optimize.basinhopping`.
        init_args (tuple): arguments for initializing the linear quantization parameters.
          Refer to `init_linear_quant_params` for more details.
        minimizer_kwargs (dict): the kwargs for scipy.optimize.minimize procedure.
        test_fn (callable): a function to test the current performance of the model.
    """
    if fold_sequences:
        model = fold_batch_norms(model, sample_input)
    if args is None:
        args = get_default_args()
    elif isinstance(args, dict):
        updated_args = get_default_args()
        updated_args.__dict__.update(args)
        args = updated_args
    original_model = deepcopy(model)
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
            args.qe_stats_file = act_stats_path
    msglogger.info('Evaluating baseline score for model...')
    base_score = args.base_score or eval_fn(model)
    msglogger.info("Baseline score: %.3f" % base_score)
    # Preparing model and init conditions:
    msglogger.info("Evaluating initial quantization score...")
    if test_fn:
        l_top1, l_top5, l_loss = test_fn(model)
        msglogger.info('Test: \tloss=%.3f, top1=%.3f, top5=%.3f ' % (l_loss, l_top1, l_top5))

    quantizer = PostTrainLinearQuantizer.from_args(model, args)

    if len(args.search_on) == 0:
        args.search_on = ['scale', 'zero_point']
    init_args = init_args or (args.init_mode, args.init_mode_method)

    quantizer.prepare_model(sample_input)
    init_linear_quant_params(quantizer, original_model, sample_input, *init_args, search_on=args.search_on)
    best_data = {
        'score': eval_fn(model),
        'qp_dict': deepcopy(quantizer.linear_quant_params)
    }
    msglogger.info("Initial quantization score %.3f" % best_data['score'])
    if test_fn:
        l_top1, l_top5, l_loss = test_fn(quantizer.model)
        msglogger.info('Test: \tloss=%.3f, top1=%.3f, top5=%.3f ' % (l_loss, l_top1, l_top5))

    init_qp_dict = deepcopy(quantizer.linear_quant_params)
    # filter buffers by the choices:
    init_qp_dict = OrderedDict((k, v) for k, v in init_qp_dict.items() if
                               any(b in k for b in args.search_on))
    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict)
    _iter = count(0)

    def feed_forward_fn(qp_vec):
        qp_dict = quant_params_vec2dict(keys, qp_vec)
        quantizer.update_linear_quant_params(qp_dict)
        return eval_fn(quantizer.model)

    def callback(qp_vec):
        score = feed_forward_fn(qp_vec)
        i = next(_iter)
        msglogger.info("Step %d: \t Score=%.3f" % (i, score))
        if score < best_data['score']:
            best_data['score'] = score
            best_data['qp_dict'] = quant_params_vec2dict(keys, qp_vec)
            msglogger.info("Saving current best quantization parameters.")
        if test_fn:
            l_top1, l_top5, l_loss = test_fn(quantizer.model)
            msglogger.info('Test: \tloss=%.3f, top1=%.3f, top5=%.3f ' % (l_loss, l_top1, l_top5))
    options = options or OrderedDict()
    if args.maxiter is not None:
        options['maxiter'] = args.maxiter
    if args.maxfev is not None:
        options['maxfev'] = args.maxfev
    minimizer_kwargs = minimizer_kwargs or OrderedDict()
    minimizer_kwargs.update({
        'method': method, 'options': options
    })
    basinhopping = basinhopping or args.basinhopping
    if basinhopping:
        msglogger.info('Using basinhopping global minimum search with "%s" local minimization method'%
                       method)
        res = opt.basinhopping(feed_forward_fn, init_qp_vec, args.niter, callback=callback,
                               minimizer_kwargs=minimizer_kwargs)
    else:
        msglogger.info('Using "%s" minimization algorithm.' % method)
        res = opt.minimize(feed_forward_fn, init_qp_vec, callback=callback, **minimizer_kwargs)

    msglogger.info("Optimization done. Best configuration: %s" % best_data['qp_dict'])
    return model, best_data['qp_dict']


if __name__ == "__main__":
    args = get_default_args()
    args.epochs = float('inf')  # hack for args parsing so there's no error in epochs
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    args = deepcopy(cc.args)
    args.effective_test_size = args.opt_test_size
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False, fixed_subset=True)

    # quant calibration dataloader:
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, fixed_subset=True, load_train=False, load_val=False)
    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)
    eval_counter = count(0)

    def eval_fn(model):
        top1, top5, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        i = next(eval_counter)
        if i % 20 == 0:
            msglogger.info('%d evaluations: loss=%.3f, top1=%.3f, top5=%.3f ' % (i, losses, top1, top5))
        return losses

    def calib_eval_fn(model):
        classifier.test(calib_data_loader, model, cc.criterion, [], None,
                        args)

    def test_fn(model):
        test_data_loader = classifier.load_data(args, load_val=False, load_train=False)
        return classifier.test(test_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                args)


    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    args.device = next(model.parameters()).device
    if args.resumed_checkpoint_path:
        args.load_model_path = args.resumed_checkpoint_path
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
    model, qp_dict = ptq_coordinate_search(model, dummy_input, eval_fn,
                                           args.method, args=args, act_stats=act_stats,
                                           calib_eval_fn=calib_eval_fn, test_fn=test_fn)
    top1, top5, loss = test_fn(model)
    msglogger.info("Test: \n\t top1 = %.3f \t top5 = %.3f \t loss = %.3f" %
                   (top1, top5, loss))
    distiller.yaml_ordered_save('%s.quant_params_dict.yaml' % args.arch, qp_dict)

