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

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input_var)
        loss = criterion(output, target_var)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import argparse
import time
import os
import sys
import random
import traceback
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger, ActivationSparsityCollector
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model


# Logger handle
msglogger = None


def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val


parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
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
parser.add_argument('--act-stats', dest='activation_stats', action='store_true', default=False,
                    help='collect activation statistics (WARNING: this slows down training)')
parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                    help='log the paramter tensors histograms to file (WARNING: this can use significant disk space)')
SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                    help='test the sensitivity of layers to pruning')
parser.add_argument('--extras', default=None, type=str,
                    help='file with extra configuration information')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--quantize', action='store_true',
                    help='Apply 8-bit quantization to model before evaluation')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true', help='Display the confusion matrix')
parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None, help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None, help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')


def check_pytorch_version():
    if torch.__version__ < '0.4.0':
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 0.4.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 0.4.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


def main():
    global msglogger
    check_pytorch_version()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(sys.argv, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    best_top1 = 0

    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        if args.workers > 1:
            msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
            exit(1)
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        cudnn.deterministic = True
    else:
        # This issue: https://github.com/pytorch/pytorch/issues/3659
        # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
        # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
        cudnn.benchmark = True

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch, device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    if args.ADC:
        HAVE_GYM_INSTALLED = False
        if not HAVE_GYM_INSTALLED:
            raise ValueError("ADC is currently experimental and uses non-public Coach features")

        import examples.automated_deep_compression.ADC as ADC
        train_loader, val_loader, test_loader, _ = apputils.load_data(
            args.dataset, os.path.expanduser(args.data), args.batch_size,
            args.workers, args.validation_size, args.deterministic)

        validate_fn = partial(validate, val_loader=test_loader, criterion=criterion,
                              loggers=[pylogger], print_freq=args.print_freq)

        save_checkpoint_fn = partial(apputils.save_checkpoint, arch=args.arch, name='adc')
        ADC.do_adc(model, args.dataset, args.arch, val_loader, validate_fn, save_checkpoint_fn)
        exit()

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        which_summary = args.summary
        if which_summary.startswith('png'):
            apputils.draw_img_classifier_to_file(model, 'model.png', args.dataset, which_summary == 'png_w_params')
        else:
            distiller.model_summary(model, which_summary, args.dataset)
        exit()

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    activations_sparsity = None
    if args.activation_stats:
        # If your model has ReLU layers, then those layers have sparse activations.
        # ActivationSparsityCollector will collect information about this sparsity.
        # WARNING! Enabling activation sparsity collection will significantly slow down training!
        activations_sparsity = ActivationSparsityCollector(model)

    if args.sensitivity is not None:
        # This sample application can be invoked to execute Sensitivity Analysis on your
        # model.  The ouptut is saved to CSV and PNG.
        msglogger.info("Running sensitivity tests")
        test_fnc = partial(test, test_loader=test_loader, criterion=criterion,
                           loggers=[pylogger], print_freq=args.print_freq)
        which_params = [param_name for param_name, _ in model.named_parameters()]
        sensitivity = distiller.perform_sensitivity_analysis(model,
                                                             net_params=which_params,
                                                             sparsities=np.arange(0.0, 0.95, 0.05),
                                                             test_func=test_fnc,
                                                             group=args.sensitivity)
        distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
        distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')
        exit()

    if args.evaluate:
        # This sample application can be invoked to evaluate the accuracy of your model on
        # the test dataset.
        # You can optionally quantize the model to 8-bit integer before evaluation.
        # For example:
        # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume=checkpoint.pth.tar --evaluate
        if args.quantize:
            model.cpu()
            quantizer = quantization.SymmetricLinearQuantizer(model, 8, 8)
            quantizer.prepare_model()
            model.cuda()
        top1, _, _ = test(test_loader, model, criterion, [pylogger], args.print_freq)
        if args.quantize:
            checkpoint_name = 'quantized'
            apputils.save_checkpoint(0, args.arch, model, optimizer=None, best_top1=top1,
                                     name='_'.split(args.name, checkpoint_name) if args.name else checkpoint_name,
                                     dir=msglogger.logdir)
        exit()

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
              loggers=[tflogger, pylogger], args=args)
        distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
        if args.activation_stats:
            distiller.log_activation_sparsity(epoch, loggers=[tflogger, pylogger],
                                              collector=activations_sparsity)

        # evaluate on validation set
        top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)
        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1, loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # remember best top1 and save checkpoint
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        apputils.save_checkpoint(epoch, args.arch, model, optimizer, compression_scheduler, best_top1, is_best,
                                 args.name, msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], args=args)


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training loop for one epoch."""
    losses = {'objective_loss':   tnt.AverageValueMeter(),
              'regularizer_loss': tnt.AverageValueMeter()}
    if compression_scheduler is None:
        # Initialize the regularizer loss to zero
        losses['regularizer_loss'].add(0)

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()
    if args.earlyexit_lossweights:
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)

        target = target.cuda(async=True)
        input_var = inputs.cuda()
        target_var = torch.autograd.Variable(target)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        output = model(input_var)
        if not args.earlyexit_lossweights:
            loss = criterion(output, target_var)
            # Measure accuracy and record loss
            classerr.add(output.data, target)
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target_var, criterion, args)

        losses['objective_loss'].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we add any regularization loss computed by the scheduler
            regularizer_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss, optimizer)
            loss += regularizer_loss
            losses['regularizer_loss'].add(regularizer_loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            lr = optimizer.param_groups[0]['lr']
            if not args.earlyexit_lossweights:
                stats = ('Peformance/Training/',
                         OrderedDict([
                             ('Loss', losses['objective_loss'].mean),
                             ('Reg Loss', losses['regularizer_loss'].mean),
                             ('Top1', classerr.value(1)),
                             ('Top5', classerr.value(5)),
                             ('LR', lr),
                             ('Time', batch_time.mean)]))
            else:
                stats_dict = OrderedDict()
                stats_dict['Objective Loss'] = losses['objective_loss'].mean
                for exitnum in range(args.num_exits):
                    t1 = 'Top1_exit' + str(exitnum)
                    t5 = 'Top5_exit' + str(exitnum)
                    stats_dict[t1] = args.exiterrors[exitnum].value(1)
                    stats_dict[t5] = args.exiterrors[exitnum].value(5)
                stats = ('Peformance/Training/', stats_dict)

            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    return _validate(test_loader, model, criterion, loggers, args)


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if args.earlyexit_thresholds:
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exitstats = [0] * args.num_exits
    
    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with PytorchNoGrad():
            target = target.cuda(async=True)
            input_var = get_inference_var(inputs)
            target_var = get_inference_var(target)
            # compute output from model
            output = model(input_var)

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target_var)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                # If using Early Exit, then compute outputs at all exits - output is now a list of all exits
                # from exit0 through exitN (i.e. [exit0, exit1, ... exitN])
                earlyexit_validate_loss(output, target_var, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                if not args.earlyexit_thresholds:
                    stats = ('',
                         OrderedDict([('Loss', losses['objective_loss'].mean),
                                      ('Top1', classerr.value(1)),
                                      ('Top5', classerr.value(5))]))
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                        # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                        # accordingly and we will not print for an exit error when that exit is never taken.
                        if args.exitstats[exitnum]:
                            t1 = 'Top1_exit' + str(exitnum)
                            t5 = 'Top5_exit' + str(exitnum)
                            stats_dict[t1] = args.exiterrors[exitnum].value(1)
                            stats_dict[t5] = args.exiterrors[exitnum].value(5)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    elif args.earlyexit_thresholds and args.num_exits == 2:
        #print some interesting summary stats for number of data points that could exit early
        msglogger.info("Exit 0: %d", args.exitstats[0])
        msglogger.info("Exit N: %d", args.exitstats[1])
        msglogger.info("Percent Early Exit #0: %.3f", (args.exitstats[0]*100.0) / (args.exitstats[0]+args.exitstats[1]))
        validate_return = [0, 0, 0]
        if args.exitstats[0]:
            validate_return[0] += args.exiterrors[0].value(1)
            validate_return[1] += args.exiterrors[0].value(5)
            validate_return[2] += args.losses_exits[0].mean
        if args.exitstats[1]:
            validate_return[0] += args.exiterrors[1].value(1)
            validate_return[1] += args.exiterrors[1].value(5)
            validate_return[2] += args.losses_exits[1].mean
        return validate_return[0], validate_return[1], validate_return[2]
    else:    # EarlyExit & imagenet
        #print some interesting summary stats for number of data points that could exit early
        msglogger.info("Exit 0: %d", args.exitstats[0])
        msglogger.info("Exit 1: %d", args.exitstats[1])
        msglogger.info("Exit N: %d", args.exitstats[2])
        msglogger.info("Percent Early Exit #0: %.3f", (args.exitstats[0]*100.0) / (args.exitstats[0]+args.exitstats[1]+args.exitstats[2]))
        msglogger.info("Percent Early Exit #1: %.3f", (args.exitstats[1]*100.0) / (args.exitstats[0]+args.exitstats[1]+args.exitstats[2]))
        validate_return = [0, 0, 0]
        if args.exitstats[0]:
            validate_return[0] += args.exiterrors[0].value(1)
            validate_return[1] += args.exiterrors[0].value(5)
            validate_return[2] += args.losses_exits[0].mean
        if args.exitstats[1]:
            validate_return[0] += args.exiterrors[1].value(1)
            validate_return[1] += args.exiterrors[1].value(5)
            validate_return[2] += args.losses_exits[1].mean
        if args.exitstats[2]:
            validate_return[0] += args.exiterrors[2].value(1)
            validate_return[1] += args.exiterrors[2].value(5)
            validate_return[2] += args.losses_exits[2].mean
        return validate_return[0], validate_return[1], validate_return[2]



class PytorchNoGrad(object):
    """This is a temporary class to bridge some difference between PyTorch 3.x and 4.x"""
    def __init__(self):
        self.no_grad = None
        if torch.__version__ >= '0.4':
            self.no_grad = torch.no_grad()

    def __enter__(self):
        if self.no_grad:
            return self.no_grad.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.no_grad:
            return self.no_grad.__exit__(self, exc_type, exc_val, exc_tb)


def get_inference_var(tensor):
    """This is a temporary function to bridge some difference between PyTorch 3.x and 4.x"""
    tensor = tensor.cuda(async=True)
    if torch.__version__ >= '0.4':
        return torch.autograd.Variable(tensor)
    return torch.autograd.Variable(tensor, volatile=True)

def earlyexit_loss(output, target_var, criterion, args):
    loss = 0
    sum_lossweights = 0
    for exitnum in range(args.num_exits-1):
        loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target_var))
        sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, target_var)
    # handle final exit
    loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target_var)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].data, target_var)
    return loss

def earlyexit_validate_loss(output, target_var, criterion, args):
    for exitnum in range(args.num_exits):
        args.loss_exits[exitnum] = criterion(output[exitnum], target_var)
        args.losses_exits[exitnum].add(args.loss_exits[exitnum].item())

    # We need to go through this batch itself - this is now a vector of losses through the batch.
    # Collecting stats on which exit early can be done across the batch at this time.
    # Note that we can't use batch_size as last batch might be smaller
    this_batch_size = target_var.size()[0]
    for batchnum in range(this_batch_size):
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits-1):
            if args.loss_exits[exitnum].item() < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batchnum], ndmin=2)),
                        torch.full([1], target_var[batchnum], dtype=torch.long))
                args.exitstats[exitnum] += 1
            else:
                # skip the early exits and include results from end of net
                args.exiterrors[args.num_exits-1].add(torch.tensor(np.array(output[args.num_exits-1].data[batchnum], ndmin=2)),
                        torch.full([1], target_var[batchnum], dtype=torch.long))
                args.exitstats[args.num_exits-1] += 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            msglogger.error(traceback.format_exc())
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
