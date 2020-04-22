import distiller
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import distiller.apputils.image_classifier as classifier
import argparse
import distiller.models as models
import logging

from examples.classifier_compression import parser
from distiller.data_loggers import *
from distiller.models import create_model
from distiller.apputils import image_classifier
from distiller.apputils import load_data
from distiller.utils import float_range_argparse_checker as float_range

def init_classifier_compression_arg_parser(include_ptq_lapq_args=False):
    '''Common classifier-compression application command-line arguments.
    '''
    SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']

    parser = argparse.ArgumentParser(description='Distiller image classification model compression')
    parser.add_argument('data', metavar='DATASET_DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', type=lambda s: s.lower(),
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, metavar='N', default=90,
                        help='number of total epochs to run (default: 90')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    load_checkpoint_group_exc.add_argument('--resume', dest='deprecated_resume', default='', type=str,
                        metavar='PATH', help=argparse.SUPPRESS)
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path',
                        default='', type=str, metavar='PATH',
                        help='path to checkpoint to load weights from (excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true',
                        help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser.add_argument('--activation-histograms', '--act-hist',
                        type=float_range(exc_min=True),
                        metavar='PORTION_OF_TEST_SET',
                        help='Run the model in evaluation mode on the specified portion of the test dataset and '
                             'generate activation histograms. NOTE: This slows down evaluation significantly')
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES, action='append',
                        help='print a summary of the model, and exit - options: | '.join(SUMMARY_CHOICES))
    parser.add_argument('--export-onnx', action='store', nargs='?', type=str, const='model.onnx', default=None,
                        help='export model to ONNX format')
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(), help='test the sensitivity of layers to pruning')
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing '
                             'providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')
    parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')
    parser.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=include_ptq_lapq_args)
    return parser

def main():
    yaml_path = '/home/dwkim/Project/distiller/examples/quantization/quant_aware_train/preact_resnet20_cifar_pact.yaml'
    model = create_model(False, 'cifar10', 'preact_resnet20_cifar',
                         parallel=not False, device_ids=None)
    print(model)

    compression_scheduler = None
    optimizer = None
    start_epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=0.0002)

    compression_scheduler = distiller.file_config(model, optimizer, yaml_path, compression_scheduler, None)

    loader = load_data('cifar10', '../../../data.cifar10', batch_size=128, workers=1)
    (train_loader, val_loader, test_loader) = (loader[0], loader[1], loader[2])
    print(model)

    criterion = nn.CrossEntropyLoss()

    # Logger handle


    # args = init_classifier_compression_arg_parser(True)
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    args.arch = 'preact_resnet20_cifar'
    args.device = 'cuda'
    args.print_freq = 50
    args.batch_size = 128
    args.log_params_histograms = False
    # args.data = '../../../data.cifar10/'
    args.dataset = 'cifar10'
    args.qe_bits_accum = 32
    args.qe_bits_acts = 8
    args.qe_bits_wts = 8
    args.qe_calibration = None

    msglogger = logging.getLogger()
    logdir = image_classifier._init_logger(args, '/home/dwkim/Project/distiller/tests/wook')
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)
    loggers = [tflogger, pylogger]

    activations_collectors = image_classifier.create_activation_stats_collectors(model, args.activation_stats)
    for epoch in range(0, 100):
        with collectors_context(activations_collectors["train"]) as collectors:
            image_classifier.train(train_loader, model, criterion, optimizer, epoch=1, compression_scheduler=compression_scheduler, loggers=loggers, args=args)



if __name__ == '__main__':
    main()
