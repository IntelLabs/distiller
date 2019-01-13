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

import argparse

import distiller
import models


SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']

def getParser():
    parser = argparse.ArgumentParser(description='Distiller image classification model compression')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', type=lambda s: s.lower(),
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        # choices=["train", "valid", "test"]
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file (WARNING: this can use significant disk space)')
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES,
                        help='print a summary of the model, and exit - options: ' +
                        ' | '.join(SUMMARY_CHOICES))
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'], type=lambda s: s.lower(),
                        help='test the sensitivity of layers to pruning')
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing providing the range of sparsities to test.\n'
                        'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser.add_argument('--extras', default=None, type=str,
                        help='file with extra configuration information')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
    parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
    parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --earlyexit_lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')

    str_to_quant_mode_map = {
                          'sym': distiller.quantization.LinearQuantMode.SYMMETRIC,
                          'asym_s': distiller.quantization.LinearQuantMode.ASYMMETRIC_SIGNED,
                          'asym_u': distiller.quantization.LinearQuantMode.ASYMMETRIC_UNSIGNED,
                          }

    def linear_quant_mode_str(val_str):
        try:
            return str_to_quant_mode_map[val_str]
        except KeyError:
            raise argparse.ArgumentError(
                    'Must be one of {0} (received {1})'.format(
                      list(str_to_quant_mode_map), val_str))

    quant_group = parser.add_argument_group('Arguments controlling quantization at evaluation time'
                                            '("post-training quantization)')
    quant_group.add_argument('--quantize-eval', '--qe', action='store_true',
                             help='Apply linear quantization to model before evaluation. Applicable only if'
                                 '--evaluate is also set')
    quant_group.add_argument('--qe-mode', '--qem', type=linear_quant_mode_str, default='sym',
                             help='Linear quantization mode. Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    quant_group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                             help='Number of bits for quantization of activations')
    quant_group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                             help='Number of bits for quantization of weights')
    quant_group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                             help='Number of bits for quantization of the accumulator')
    quant_group.add_argument('--qe-clip-acts', '--qeca', action='store_true',
                             help='Enable clipping of activations using min/max values averaging over batch')
    quant_group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                             help='List of layer names for which not to clip activations. Applicable only if '
                                  '--qe-clip-acts is also set')
    quant_group.add_argument('--qe-per-channel', '--qepc', action='store_true',
                             help='Enable per-channel quantization of weights (per output channel)')

    return parser


def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val
