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

import copy
import math
import time
import os
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
import parser
from functools import partial
import argparse
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
import distiller.models as models
from distiller.models import create_model
from distiller.utils import float_range_argparse_checker as float_range

# Logger handle
msglogger = logging.getLogger()


class ClassifierCompressor(object):
    """Base class for applications that want to compress image classifiers.

    This class performs boiler-plate code used in image-classifiers compression:
        - Command-line arguments handling
        - Logger configuration
        - Data loading
        - Checkpoint handling
        - Classifier training, verification and testing
    """
    def __init__(self, args, script_dir):
        self.args = copy.deepcopy(args)
        self._infer_implicit_args(self.args)
        self.logdir = _init_logger(self.args, script_dir)
        _config_determinism(self.args)
        _config_compute_device(self.args)
        
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
        if not self.logdir:
            self.pylogger = self.tflogger = NullLogger()
        else:
            self.tflogger = TensorBoardLogger(msglogger.logdir)
            self.pylogger = PythonLogger(msglogger)
        (self.model, self.compression_scheduler, self.optimizer, 
             self.start_epoch, self.ending_epoch) = _init_learner(self.args)

        # Define loss function (criterion)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.train_loader, self.val_loader, self.test_loader = (None, None, None)
        self.activations_collectors = create_activation_stats_collectors(
            self.model, *self.args.activation_stats)
        self.performance_tracker = apputils.SparsityAccuracyTracker(self.args.num_best_scores)
    
    def load_datasets(self):
        """Load the datasets"""
        if not all((self.train_loader, self.val_loader, self.test_loader)):
            self.train_loader, self.val_loader, self.test_loader = load_data(self.args)
        return self.data_loaders

    @property
    def data_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    @staticmethod
    def _infer_implicit_args(args):
        # Infer the dataset from the model name
        if not hasattr(args, 'dataset'):
            args.dataset = distiller.apputils.classification_dataset_str_from_arch(args.arch)
        if not hasattr(args, "num_classes"):
            args.num_classes = distiller.apputils.classification_num_classes(args.dataset)
        return args

    @staticmethod
    def mock_args():
        """Generate a Namespace based on default arguments"""
        return ClassifierCompressor._infer_implicit_args(
            init_classifier_compression_arg_parser().parse_args(['fictive_required_arg',]))

    @classmethod
    def mock_classifier(cls):
        return cls(cls.mock_args(), '')

    def train_one_epoch(self, epoch, verbose=True):
        """Train for one epoch"""
        self.load_datasets()

        with collectors_context(self.activations_collectors["train"]) as collectors:
            top1, top5, loss = train(self.train_loader, self.model, self.criterion, self.optimizer, 
                                     epoch, self.compression_scheduler, 
                                     loggers=[self.tflogger, self.pylogger], args=self.args)
            if verbose:
                distiller.log_weights_sparsity(self.model, epoch, [self.tflogger, self.pylogger])
            distiller.log_activation_statistics(epoch, "train", loggers=[self.tflogger],
                                                collector=collectors["sparsity"])
            if self.args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(self.model, 
                                                                    self.compression_scheduler))
        return top1, top5, loss

    def train_validate_with_scheduling(self, epoch, validate=True, verbose=True):
        if self.compression_scheduler:
            self.compression_scheduler.on_epoch_begin(epoch)

        top1, top5, loss = self.train_one_epoch(epoch, verbose)
        if validate:
            top1, top5, loss = self.validate_one_epoch(epoch, verbose)

        if self.compression_scheduler:
            self.compression_scheduler.on_epoch_end(epoch, self.optimizer, 
                                                    metrics={'min': loss, 'max': top1})
        return top1, top5, loss

    def validate_one_epoch(self, epoch, verbose=True):
        """Evaluate on validation set"""
        self.load_datasets()
        with collectors_context(self.activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(self.val_loader, self.model, self.criterion, 
                                         [self.pylogger], self.args, epoch)
            distiller.log_activation_statistics(epoch, "valid", loggers=[self.tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        if verbose:
            stats = ('Performance/Validation/',
            OrderedDict([('Loss', vloss),
                         ('Top1', top1),
                         ('Top5', top5)]))
            distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=[self.tflogger])
        return top1, top5, vloss

    def _finalize_epoch(self, epoch, top1, top5):
        # Update the list of top scores achieved so far, and save the checkpoint
        self.performance_tracker.step(self.model, epoch, top1=top1, top5=top5)
        _log_best_scores(self.performance_tracker, msglogger)
        best_score = self.performance_tracker.best_scores()[0]
        is_best = epoch == best_score.epoch
        checkpoint_extras = {'current_top1': top1,
                             'best_top1': best_score.top1,
                             'best_epoch': best_score.epoch}
        if msglogger.logdir:
            apputils.save_checkpoint(epoch, self.args.arch, self.model, optimizer=self.optimizer,
                                     scheduler=self.compression_scheduler, extras=checkpoint_extras,
                                     is_best=is_best, name=self.args.name, dir=msglogger.logdir)

    def run_training_loop(self):
        """Run the main training loop with compression.

        For each epoch:
            train_one_epoch
            validate_one_epoch
            finalize_epoch
        """
        if self.start_epoch >= self.ending_epoch:
            msglogger.error(
                'epoch count is too low, starting epoch is {} but total epochs set to {}'.format(
                self.start_epoch, self.ending_epoch))
            raise ValueError('Epochs parameter is too low. Nothing to do.')

        # Load the datasets lazily
        self.load_datasets()

        self.performance_tracker.reset()
        for epoch in range(self.start_epoch, self.ending_epoch):
            msglogger.info('\n')
            top1, top5, loss = self.train_validate_with_scheduling(epoch)
            self._finalize_epoch(epoch, top1, top5)
        return self.performance_tracker.perf_scores_history

    def validate(self, epoch=-1):
        self.load_datasets()
        return validate(self.val_loader, self.model, self.criterion,
                        [self.tflogger, self.pylogger], self.args, epoch)

    def test(self):
        self.load_datasets()
        return test(self.test_loader, self.model, self.criterion,
                    self.pylogger, self.activations_collectors, args=self.args)


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


def _init_logger(args, script_dir):
    global msglogger
    if script_dir is None or not hasattr(args, "output_dir") or args.output_dir is None:
        msglogger.logdir = None
        return None
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'),
                                         args.name, args.output_dir, args.verbose)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)
    return msglogger.logdir


def _config_determinism(args):
    if args.evaluate:
        args.deterministic = True
    
    # Configure some seed (in case we want to reproduce this experiment session)
    if args.seed is None:
        if args.deterministic:
            args.seed = 0
        else:
            args.seed = np.random.randint(1, 100000)

    if args.deterministic:
        distiller.set_deterministic(args.seed) # For experiment reproducability
    else:
        distiller.set_seed(args.seed)
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True
    msglogger.info("Random seed: %d", args.seed)


def _config_compute_device(args):
    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])


def _init_learner(args):
    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    compression_scheduler = None

    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    if args.deprecated_resume:
        msglogger.warning('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            msglogger.warning('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    optimizer = None
    start_epoch = 0
    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    if optimizer is None and not args.evaluate:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        msglogger.debug('Optimizer Type: %s', type(optimizer))
        msglogger.debug('Optimizer Args: %s', optimizer.defaults)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) if args.resumed_checkpoint_path else None)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    return model, compression_scheduler, optimizer, start_epoch, args.epochs


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    genCollectors = lambda: missingdict({
        "sparsity_ofm":      SummaryActivationStatsCollector(model, "sparsity_ofm",
            lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to disk.

    File type and format of contents are collector-specific.
    """
    for name, collector in collectors.items():
        msglogger.info('Saving data for collector {}...'.format(name))
        file_path = collector.save(os.path.join(directory, name))
        msglogger.info("Saved to {}".format(file_path))


def load_data(args, fixed_subset=False, sequential=False, load_train=True, load_val=True, load_test=True):
    test_only = not load_train and not load_val

    train_loader, val_loader, test_loader, _ = apputils.load_data(args.dataset,
                              os.path.expanduser(args.data), args.batch_size,
                              args.workers, args.validation_split, args.deterministic,
                              args.effective_train_size, args.effective_valid_size, args.effective_test_size,
                              fixed_subset, sequential, test_only)
    if test_only:
        msglogger.info('Dataset sizes:\n\ttest=%d', len(test_loader.sampler))
    else:
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                       len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    loaders = (train_loader, val_loader, test_loader)
    flags = (load_train, load_val, load_test)
    loaders = [loaders[i] for i, flag in enumerate(flags) if flag]
    
    if len(loaders) == 1:
        # Unpack the list for convinience
        loaders = loaders[0]
    return loaders


def early_exit_mode(args):
    return hasattr(args, 'earlyexit_lossweights') and args.earlyexit_lossweights


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training-with-compression loop for one epoch.
    
    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """
    def _log_training_progress():
        # Log some statistics
        errs = OrderedDict()
        if not early_exit_mode(args):
            errs['Top1'] = classerr.value(1)
            errs['Top5'] = classerr.value(5)
        else:
            # For Early Exit case, the Top1 and Top5 stats are computed for each exit.
            for exitnum in range(args.num_exits):
                errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

        stats_dict = OrderedDict()
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict.update(errs)
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean
        stats = ('Performance/Training/', stats_dict)

        params = model.named_parameters() if args.log_params_histograms else None
        distiller.log_training_progress(stats,
                                        params,
                                        epoch, steps_completed,
                                        steps_per_epoch, args.print_freq,
                                        loggers)

    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit, so
    # `exiterrors` is analogous to `classerr` in the non-Early Exit case
    if early_exit_mode(args):
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if not hasattr(args, 'kd_policy') or args.kd_policy is None:
            output = model(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if not early_exit_mode(args):
            loss = criterion(output, target)
            # Measure accuracy
            classerr.add(output.detach(), target)
            acc_stats.append([classerr.value(1), classerr.value(5)])
        else:
            # Measure accuracy and record loss
            classerr.add(output[args.num_exits-1].detach(), target) # add the last exit (original exit)
            loss = earlyexit_loss(output, target, criterion, args)
        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            _log_training_progress()

        end = time.time()
    #return acc_stats
    # NOTE: this breaks previous behavior, which returned a history of (top1, top5) values
    return classerr.value(1), classerr.value(5), losses[OVERALL_LOSS_KEY]


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if args is None:
        args = ClassifierCompressor.mock_args()
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)

    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses


# Temporary patch until we refactor early-exit handling
def _is_earlyexit(args):
    return hasattr(args, 'earlyexit_thresholds') and args.earlyexit_thresholds


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    def _log_validation_progress():
        if not _is_earlyexit(args):
            stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),
                                      ('Top1', classerr.value(1)),
                                      ('Top5', classerr.value(5))])
        else:
            stats_dict = OrderedDict()
            for exitnum in range(args.num_exits):
                la_string = 'LossAvg' + str(exitnum)
                stats_dict[la_string] = args.losses_exits[exitnum].mean
                # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                # accordingly and we will not print for an exit error when that exit is never taken.
                if args.exit_taken[exitnum]:
                    t1 = 'Top1_exit' + str(exitnum)
                    t5 = 'Top5_exit' + str(exitnum)
                    stats_dict[t1] = args.exiterrors[exitnum].value(1)
                    stats_dict[t5] = args.exiterrors[exitnum].value(5)
        stats = ('Performance/Validation/', stats_dict)
        distiller.log_training_progress(stats, None, epoch, steps_completed,
                                        total_steps, args.print_freq, loggers)

    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if _is_earlyexit(args):
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for validation_step, (inputs, target) in enumerate(data_loader):
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            output = model(inputs)

            if not _is_earlyexit(args):
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.detach(), target)
                if args.display_confusion:
                    confusion.add(output.detach(), target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                _log_validation_progress()

    if not _is_earlyexit(args):
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


def earlyexit_loss(output, target, criterion, args):
    """Compute the weighted sum of the exits losses

    Note that the last exit is the original exit of the model (i.e. the
    exit that traverses the entire network.
    """
    weighted_loss = 0
    sum_lossweights = sum(args.earlyexit_lossweights)
    assert sum_lossweights < 1
    for exitnum in range(args.num_exits-1):
        if output[exitnum] is None:
            continue
        exit_loss = criterion(output[exitnum], target)
        weighted_loss += args.earlyexit_lossweights[exitnum] * exit_loss
        args.exiterrors[exitnum].add(output[exitnum].detach(), target)
    # handle final exit
    weighted_loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].detach(), target)
    return weighted_loss


def earlyexit_validate_loss(output, target, criterion, args):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batch size of 1,
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    this_batch_size = target.size(0)
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)

    for exitnum in range(args.num_exits):
        # calculate losses at each sample separately in the minibatch.
        args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits - 1):
            if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break                    # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                         torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1


def earlyexit_validate_stats(args):
    # Print some interesting summary stats for number of data points that could exit early
    top1k_stats = [0] * args.num_exits
    top5k_stats = [0] * args.num_exits
    losses_exits_stats = [0] * args.num_exits
    sum_exit_stats = 0
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            sum_exit_stats += args.exit_taken[exitnum]
            msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
            top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
            top5k_stats[exitnum] += args.exiterrors[exitnum].value(5)
            losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    total_top1 = 0
    total_top5 = 0
    for exitnum in range(args.num_exits):
        total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum, top1k_stats[exitnum], top5k_stats[exitnum])
    msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f", total_top1, total_top5)
    return total_top1, total_top5, losses_exits_stats


def _convert_ptq_to_pytorch(model, args):
    msglogger.info('Converting Distiller PTQ model to PyTorch quantization API')
    dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
    model = quantization.convert_distiller_ptq_model_to_pytorch(model, dummy_input, backend=args.qe_pytorch_backend)
    msglogger.debug('\nModel after conversion:\n{}'.format(model))
    args.device = 'cpu'
    return model


def evaluate_model(test_loader, model, criterion, loggers, activations_collectors=None, args=None, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if not args.quantize_eval:
        # Handle case where a post-train quantized model was loaded, and user wants to convert it to PyTorch
        if args.qe_convert_pytorch:
            model = _convert_ptq_to_pytorch(model, args)
        return test(test_loader, model, criterion, loggers, activations_collectors, args=args)
    else:
        return quantize_and_test_model(test_loader, model, criterion, args, loggers,
                                       scheduler=scheduler, save_flag=True)


def quantize_and_test_model(test_loader, model, criterion, args, loggers=None, scheduler=None, save_flag=True):
    """Collect stats using test_loader (when stats file is absent),

    clone the model and quantize the clone, and finally, test it.
    args.device is allowed to differ from the model's device.
    When args.qe_calibration is set to None, uses 0.05 instead.

    scheduler - pass scheduler to store it in checkpoint
    save_flag - defaults to save both quantization statistics and checkpoint.
    """
    if hasattr(model, 'quantizer_metadata') and \
            model.quantizer_metadata['type'] == distiller.quantization.PostTrainLinearQuantizer:
        raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                           'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                           'passing the --quantize-eval flag')
    if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
        args_copy = copy.deepcopy(args)
        args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

        # set stats into args stats field
        args.qe_stats_file = acts_quant_stats_collection(
            model, criterion, loggers, args_copy, save_to_file=save_flag)

    args_qe = copy.deepcopy(args)
    if args.device == 'cpu':
        # NOTE: Even though args.device is CPU, we allow here that model is not in CPU.
        qe_model = distiller.make_non_parallel_copy(model).cpu()
    else:
        qe_model = copy.deepcopy(model).to(args.device)

    quantizer = quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
    dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
    quantizer.prepare_model(dummy_input)

    if args.qe_convert_pytorch:
        qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)

    test_res = test(test_loader, qe_model, criterion, loggers, args=args_qe)

    if save_flag:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
            name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
            dir=msglogger.logdir, extras={'quantized_top1': test_res[0]})

    del qe_model
    return test_res


def acts_quant_stats_collection(model, criterion, loggers, args, test_loader=None, save_to_file=False):
    msglogger.info('Collecting quantization calibration stats based on {:.1%} of test dataset'
                   .format(args.qe_calibration))
    if test_loader is None:
        tmp_args = copy.deepcopy(args)
        tmp_args.effective_test_size = tmp_args.qe_calibration
        # Batch size 256 causes out-of-memory errors on some models (due to extra space taken by
        # stats calculations). Limiting to 128 for now.
        # TODO: Come up with "smarter" limitation?
        tmp_args.batch_size = min(128, tmp_args.batch_size)
        test_loader = load_data(tmp_args, fixed_subset=True, load_train=False, load_val=False)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    with distiller.get_nonparallel_clone_model(model) as cmodel:
        return collect_quant_stats(cmodel, test_fn, classes=None,
                                   inplace_runtime_check=True, disable_inplace_attrs=True,
                                   save_dir=msglogger.logdir if save_to_file else None)


def acts_histogram_collection(model, criterion, loggers, args):
    msglogger.info('Collecting activation histograms based on {:.1%} of test dataset'
                   .format(args.activation_histograms))
    model = distiller.utils.make_non_parallel_copy(model)
    args.effective_test_size = args.activation_histograms
    test_loader = load_data(args, fixed_subset=True, load_train=False, load_val=False)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    collect_histograms(model, test_fn, save_dir=msglogger.logdir,
                       classes=None, nbins=2048, save_hist_imgs=True)


def _log_best_scores(performance_tracker, logger, how_many=-1):
    """Utility to log the best scores.

    This function is currently written for pruning use-cases, but can be generalized.
    """
    assert isinstance(performance_tracker, (apputils.SparsityAccuracyTracker))
    if how_many < 1:
        how_many = performance_tracker.max_len
    how_many = min(how_many, performance_tracker.max_len)
    best_scores = performance_tracker.best_scores(how_many)
    for score in best_scores:
        logger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   NNZ-Params: %d on epoch: %d]',
                    score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)
