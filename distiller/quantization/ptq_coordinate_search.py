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
from torch.utils.data import DataLoader
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, \
    RangeLinearQuantWrapper, RangeLinearEmbeddingWrapper, is_post_train_quant_wrapper, LinearQuantMode
from distiller.quantization import is_linear_quant_mode_asymmetric, is_linear_quant_mode_symmetric
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


def quant_params_dict2vec(p_dict, quant_mode=LinearQuantMode.SYMMETRIC, search_clipping=False):
    keys, vals = zip(*p_dict.items())  # unzip the list of tuples
    if search_clipping:
        if is_linear_quant_mode_symmetric(quant_mode):
            vals = np.array([val[1].item() for val in vals])  # only take the max_val
        else:
            vals = np.concatenate([(v1.item(), v2.item()) for v1, v2 in vals])  # stack the clipping values
    else:
        vals = np.array([val.item() for val in vals])

    return keys, vals


def quant_params_vec2dict(keys, vals, quant_mode=LinearQuantMode.SYMMETRIC, search_clipping=False):
    if not search_clipping or is_linear_quant_mode_symmetric(quant_mode):
        if vals.size == 1:
            vals = vals.reshape((1,))
        return OrderedDict(zip(keys, (val.item() for val in vals)))
    else:
        # In this case - we have N keys but 2N values. for that we reshape
        # the array to match the size: keys:N, vals: Nx2
        vals = vals.reshape(-1, 2)
        return OrderedDict(zip(keys, vals))


def lp_loss(x, y, p):
    loss = (torch.sum(torch.abs(x - y) ** p) / x.numel()).item()
    return loss


def _check_qp_vec(keys, qp_vec, quant_mode=LinearQuantMode.SYMMETRIC, search_clipping=False):
    if is_linear_quant_mode_symmetric(quant_mode):
        return all(qp_vec > 0)
    if not search_clipping:
        idxs_scales = np.array(['scale' in key for key in keys])
        qp_vec_scales = qp_vec[idxs_scales]
        return all(qp_vec_scales > 0)


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
    search_clipping = 'clipping' in search_on
    quant_mode = quantized_layer.output_quant_settings.quant_mode
    params_gen = quantized_layer.named_linear_quant_params() if not search_clipping\
        else quantized_layer.named_clipping()
    init_qp_dict = OrderedDict(params_gen)
    init_qp_dict = OrderedDict((k, v) for k, v in init_qp_dict.items() if
                               any(b in k for b in search_on))

    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict, quant_mode, search_clipping)

    def feed_forward_fn(qp_vec):
        if search_clipping:
            qp_vec.sort()
        qp_dict = quant_params_vec2dict(keys, qp_vec, quant_mode, search_clipping)
        quantized_layer.update_linear_quant_params(qp_dict)
        # Using cloned input, required if the layer is inplace
        y = layer(input.clone().detach())
        if getattr(quantized_layer, 'clip_half_range', False):
            torch.relu_(y)
        q_y = quantized_layer(input.clone().detach())
        loss = loss_fn(y, q_y)
        return loss

    result = opt.minimize(feed_forward_fn, init_qp_vec, method=method)  # type: opt.OptimizeResult
    qp_dict = quant_params_vec2dict(keys, result.x, quant_mode, search_clipping)
    quantized_layer.update_linear_quant_params(qp_dict)
    return quantized_layer


def get_input_for_layer(model, layer_name, eval_fn):
    layer = dict(model.named_modules())[layer_name]  # type: nn.Module
    layer_inputs = []

    def hook_layer_input(module, input):
        layer_inputs.append(input[0].clone().detach())

    handle = layer.register_forward_pre_hook(hook_layer_input)
    eval_fn(model)
    assert len(layer_inputs) == 1
    handle.remove()
    return layer_inputs[0]


def init_layer_linear_quant_params(quantizer, original_model, layer_name, init_mode,
                                   init_mode_method='Powell', eval_fn=None, search_on=[]):
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
        eval_fn: evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
          applicable only in the case of init_mode = 'L1/2/3' or callable.
        search_on (list): list of quant_param names to search on. Choices : 'scale' and 'zero_point'.
    """
    denorm_layer_name = distiller.denormalize_module_name(quantizer.model, layer_name)
    if isinstance(init_mode, str):
        init_mode = _init_mode_from_str(init_mode)
    if isinstance(init_mode, ClipMode):
        quantizer.module_overrides_map[denorm_layer_name]['clip_acts'] = init_mode
    layer = dict(original_model.named_modules())[layer_name]
    local_args, local_kwargs = quantizer.modules_processed_args[denorm_layer_name]
    replace_fn = quantizer.replacement_factory.get(type(layer), quantizer.default_repalcement_fn)
    quantized_layer = replace_fn(deepcopy(layer), *local_args, **local_kwargs)
    if not is_post_train_quant_wrapper(quantized_layer, False):
        return quantized_layer  # the module wasn't quantized, nothing to do here

    if callable(init_mode):
        input_for_layer = get_input_for_layer(original_model, layer_name, eval_fn)
        quantized_layer = optimize_for_layer(layer, quantized_layer, init_mode, input_for_layer, init_mode_method,
                                             search_on=search_on)

    distiller.model_setattr(quantizer.model, denorm_layer_name, quantized_layer)
    quantizer.model.eval()


def init_linear_quant_params(quantizer, original_model, eval_fn, dummy_input, init_mode,
                             init_mode_method=None, search_on=[]):
    """
    Initializes all linear quantization parameters of the model.
    Args:
        quantizer (PostTrainLinearQuantizer): the quantizer, **after** calling prepare model.
        original_model (nn.Module): the original, pre-quantized, model.
        init_mode (ClipMode or callable or str or dict): See `init_layer_linear_qaunt_params`.
          if init_mode is dict - init_mode is configuration for the different layers,
          i.e. init_mode = Dict[layer_name:str, init_mode_layer: ClipMode or callable or str].
        eval_fn: evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
          Note - unlike in `init_layer_linear_quant_params`, this argument is required here.
        dummy_input: dummy sample input to the model
        init_mode_method: See `init_layer_linear_qaunt_params`.
        search_on (list): list of quant_param names to search on. Choices : 'scale' and 'zero_point'.
    """
    original_model = distiller.make_non_parallel_copy(original_model)
    layers_topological_order = SummaryGraph(original_model, dummy_input).layers_topological_order()
    q_named_modules = OrderedDict(quantizer.model.named_modules())
    for module_name in layers_topological_order:
        # check to see if it was quantized:
        q_module = q_named_modules[distiller.denormalize_module_name(quantizer.model, module_name)]
        if not is_post_train_quant_wrapper(q_module, False):
            continue
        module_init_mode = init_mode[module_name] if isinstance(init_mode, dict) else init_mode
        msglogger.debug('Initializing layer \'%s\' using %s mode' % (module_name, module_init_mode))
        init_layer_linear_quant_params(quantizer, original_model, module_name, module_init_mode,
                                       init_mode_method=init_mode_method,
                                       eval_fn=eval_fn,
                                       search_on=search_on)
    quantizer._post_prepare_model()
    quantizer.model.eval()


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
    parser.add_argument('--opt-val-size', type=float, default=1,
                        help='Use portion of the test size.')
    parser.add_argument('--opt-eval-memoize-dataloader', dest='memoize_dataloader', action='store_true', default=False,
                        help='Stores the input batch in memory to optimize performance.')
    parser.add_argument('--base-score', type=float, default=None)
    parser.add_argument('--opt-search-clipping', dest='search_clipping', action='store_true',
                        help='Search on clipping values instead of scale/zero_point.')
    args = parser.parse_args()
    return args


def ptq_coordinate_search(model, dummy_input, eval_fn, method='Powell', options=None, calib_eval_fn=None,
                          act_stats=None, args=None, fold_sequences=True, basinhopping=False,
                          init_args=None, minimizer_kwargs=None,
                          test_fn=None):
    """
    Searches for the optimal post-train quantization configuration (scale/zero_points)
    for a model using numerical methods, as described by scipy.optimize.minimize.
    Args:
        model (nn.Module): model to quantize
        dummy_input: an sample expected input to the model
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
        model = fold_batch_norms(model, dummy_input)
    if args is None:
        args = get_default_args()
    elif isinstance(args, dict):
        updated_args = get_default_args()
        updated_args.__dict__.update(args)
        args = updated_args
    original_model = deepcopy(model)
    calib_eval_fn = calib_eval_fn or eval_fn
    if not act_stats and not args.qe_config_file:
        msglogger.info('Collecting stats for model...')
        model_temp = distiller.utils.make_non_parallel_copy(model)
        act_stats = collect_quant_stats(model_temp, calib_eval_fn)
        del model_temp
        if args:
            act_stats_path = '%s_act_stats.yaml' % args.arch
            msglogger.info('Done. Saving act stats into %s' % act_stats_path)
            distiller.yaml_ordered_save(act_stats_path, act_stats)
            args.qe_stats_file = act_stats_path
    # msglogger.info('Evaluating baseline score for model...')
    # base_score = args.base_score or eval_fn(model)
    # msglogger.info("Baseline score: %.3f" % base_score)
    # if test_fn:
    #     l_top1, l_top5, l_loss = test_fn(model)
    #     msglogger.info('Test: \tloss=%.3f, top1=%.3f, top5=%.3f ' % (l_loss, l_top1, l_top5))
    # Preparing model and init conditions:
    msglogger.info("Evaluating initial quantization score...")

    quantizer = PostTrainLinearQuantizer.from_args(model, args)

    init_args = init_args or (args.init_mode, args.init_mode_method)

    quantizer.prepare_model(dummy_input)

    assert quantizer.mode.activations == quantizer.mode.weights, 'Mixed quantization mode not supported yet'
    for n, m in quantizer.model.named_modules():
        if not is_post_train_quant_wrapper(m):
            continue
        assert m.mode == quantizer.mode, 'Overriding quantization mode not supported yet'

    if args.search_clipping:
        search_on = ['clipping']
    else:
        search_on = ['scale']
        if is_linear_quant_mode_asymmetric(quantizer.mode.activations):
            search_on.append('zero_point')

    init_linear_quant_params(quantizer, original_model, eval_fn, dummy_input, *init_args, search_on=search_on)
    best_data = {
        'score': eval_fn(model),
        'qp_dict': deepcopy(quantizer.linear_quant_params)
    }
    msglogger.info("Initial quantization score %.3f" % best_data['score'])
    if test_fn:
        l_top1, l_top5, l_loss = test_fn(quantizer.model)
        msglogger.info('Test: \tloss=%.3f, top1=%.3f, top5=%.3f ' % (l_loss, l_top1, l_top5))
    yield_clipping_params = args.search_clipping
    quant_mode = quantizer.mode.activations  # TODO: handle separate acts/wts mode
    init_qp_dict = OrderedDict(quantizer.named_linear_quant_params(yield_clipping_params))
    # filter buffers by the choices:
    init_qp_dict = OrderedDict((k, v) for k, v in init_qp_dict.items() if
                               any(b in k for b in search_on))
    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict, quant_mode, yield_clipping_params)
    _iter = count(0)

    def feed_forward_fn(qp_vec):
        if not _check_qp_vec(keys, qp_vec, quant_mode, yield_clipping_params):
            return 1e6
        qp_dict = quant_params_vec2dict(keys, qp_vec, quant_mode, yield_clipping_params)
        quantizer.update_linear_quant_params(qp_dict)
        return eval_fn(quantizer.model)

    def callback(qp_vec):
        score = feed_forward_fn(qp_vec)
        i = next(_iter)
        msglogger.info("Step %d: \t Score=%.3f" % (i, score))
        if score < best_data['score']:
            best_data['score'] = score
            best_data['qp_dict'] = quant_params_vec2dict(keys, qp_vec, quant_mode, yield_clipping_params)
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

    args.effective_valid_size = args.opt_val_size
    eval_data_loader = classifier.load_data(args, load_train=False, load_test=False, fixed_subset=True)

    # quant calibration dataloader:
    effective_test_size_bak = args.effective_test_size
    batch_size_bak = args.batch_size
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, fixed_subset=True, load_train=False, load_val=False)

    args.effective_test_size = effective_test_size_bak
    args.batch_size = batch_size_bak
    test_data_loader = classifier.load_data(args, load_train=False, load_val=False)

    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus).eval()
    device = next(model.parameters()).device
    eval_counter = count(0)

    if args.memoize_dataloader:
        memoized_data_loader = []
        for images, targets in eval_data_loader:
            batch = images.to(device), targets.to(device)
            memoized_data_loader.append(batch)
    else:
        memoized_data_loader = None

    def eval_fn(model):
        if args.memoize_dataloader:
            losses = 0
            for images, targets in memoized_data_loader:
                outputs = model(images)
                losses += cc.criterion(outputs, targets).item()
            losses = losses / len(memoized_data_loader)
        else:
            _, _, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger],
                                           None, args)
        i = next(eval_counter)
        if i % 20 == 0:
            msglogger.info('%d evaluations: loss=%.3f' % (i, losses))
        return losses

    def calib_eval_fn(model):
        classifier.test(calib_data_loader, model, cc.criterion, [], None, args)

    def test_fn(model):
        return classifier.test(test_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None, args)


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
    msglogger.info("Arch: %s \tTest: \t top1 = %.3f \t top5 = %.3f \t loss = %.3f" %
                   (args.arch, top1, top5, loss))
    distiller.yaml_ordered_save('%s.quant_params_dict.yaml' % args.arch, qp_dict)

