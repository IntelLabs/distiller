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
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, LinearQuantMode
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


def get_default_args():
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--qe-no-quant-layers', '--qenql', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                        help='List of layer names for which to skip quantization.')
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
    parser.add_argument('--opt-test-size', type=float, default=1,
                        help='Use portion of the test size.')
    parser.add_argument('--base-score', type=float, default=None)
    args = parser.parse_args()
    return args


def ptq_coordinate_search(model, dummy_input, eval_fn, method='Powell', options=None, calib_eval_fn=None,
                          act_stats=None, args=None, fold_sequences=True, basinhopping=False,
                          minimizer_kwargs=None):
    """
    Searches for the optimal post-train quantization configuration (scale/zero_points)
    for a model using numerical methods, as described by scipy.optimize.minimize.
    Args:
        model (nn.Module): model to quantize
        dummy_input: an dummy expected input to the model
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
    """
    if fold_sequences:
        model = fold_batch_norms(model, dummy_input)
    if args is None:
        args = get_default_args()
    elif isinstance(args, dict):
        updated_args = get_default_args()
        updated_args.__dict__.update(args)
        args = updated_args

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
    quantizer = PostTrainLinearQuantizer.from_args(model, args)
    quantizer.prepare_model(dummy_input)
    best_data = {
        'score': eval_fn(model),
        'qp_dict': deepcopy(quantizer.acts_quant_params)
    }
    msglogger.info("Initial quantization score %.3f" % best_data['score'])
    init_qp_dict = deepcopy(quantizer.acts_quant_params)
    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict)
    _iter = count(0)

    def feed_forward_fn(qp_vec):
        qp_dict = quant_params_vec2dict(keys, qp_vec)
        quantizer.update_acts_quant_params(qp_dict)
        return eval_fn(quantizer.model)

    def callback(qp_vec):
        score = feed_forward_fn(qp_vec)
        i = next(_iter)
        msglogger.info("Step %d: \t Score=%.3f" % (i, score))
        if score < best_data['score']:
            best_data['score'] = score
            best_data['qp_dict'] = quant_params_vec2dict(keys, qp_vec)
            msglogger.info("Saving current best quantization parameters.")
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
    args.effective_test_size = args.opt_test_size
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False, fixed_subset=True,
                                            )

    # quant calibration dataloader:
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, fixed_subset=True, load_train=False, load_val=False)
    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)
    eval_counter = count(0)

    def test_fn(model):
        top1, top5, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        i = next(eval_counter)
        if i % 20 == 0:
            msglogger.info('%d evaluations: loss=%.3f, top1=%.3f, top5=%.3f ' % (i, losses, top1, top5))
        return losses

    def calib_eval_fn(model):
        classifier.test(calib_data_loader, model, cc.criterion, [], None,
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
    model, qp_dict = ptq_coordinate_search(model, dummy_input, test_fn,
                                           args.method, args=args, act_stats=act_stats,
                                           calib_eval_fn=calib_eval_fn)
    distiller.yaml_ordered_save('%s.quant_params_dict.yaml' % args.arch, qp_dict)

