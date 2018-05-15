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
import logging.config
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
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger, ActivationSparsityCollector
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model


msglogger = None
log_filename = ''


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
SUMMARY_CHOICES = ['sparsity', 'compute', 'optimizer', 'model', 'modules', 'png']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter'],
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


def main():

    args = parser.parse_args()

    # The Distiller library writes logs to the Python logger, so we configure it.
    global msglogger
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    filename = timestr if args.name is None else args.name + '___' + timestr
    logdir = './logs' + '/' + filename
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, filename + '.log')
    logging.config.fileConfig(os.path.join(script_dir, 'logging.conf'), defaults={'logfilename': log_filename})
    msglogger = logging.getLogger()

    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

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

    # Create the model
    is_parallel = args.summary != 'png' # For PNG summary, parallel graphs are illegible
    model = create_model(args.pretrained, args.dataset, args.arch, parallel=is_parallel, device_ids=args.gpus)

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(logdir)
    pylogger = PythonLogger(msglogger)

    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)

        if 'resnet' in args.arch and 'cifar' in args.arch:
            distiller.resnet_cifar_remove_layers(model)
            #model = distiller.resnet_cifar_remove_channels(model, compression_scheduler.zeros_mask_dict)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    msglogger.info("Optimizer (%s): momentum=%s decay=%s", type(optimizer),
                   args.momentum, args.weight_decay)


    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        which_summary = args.summary
        if which_summary == 'png':
            apputils.draw_img_classifier_to_file(model, 'model.png', args.dataset)
        else:
            distiller.model_summary(model, optimizer, which_summary, args.dataset)
        exit()

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.deterministic)
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
                                                             sparsities=np.arange(0.0, 0.50, 0.05) if args.sensitivity == 'filter' else np.arange(0.0, 0.95, 0.05),
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
            apputils.save_checkpoint(0, args.arch, model, optimizer, best_top1=top1,
                                     name='_'.split(args.name, checkpoint_name) if args.name else checkpoint_name)
        exit()

    if args.compress:
        # The main use-case for this sample application is CNN compression.  Compression
        # requires a compression schedule configuration file in YAML.
        source = args.compress
        msglogger.info("Compression schedule (source=%s)", source)
        compression_scheduler = distiller.CompressionScheduler(model)
        distiller.config.fileConfig(model, optimizer, compression_scheduler, args.compress, msglogger)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
              loggers=[tflogger, pylogger], print_freq=args.print_freq, log_params_hist=args.log_params_histograms)
        distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
        if args.activation_stats:
            distiller.log_activation_sparsity(epoch, loggers=[tflogger, pylogger],
                                              collector=activations_sparsity)

        # evaluate on validation set
        top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args.print_freq, epoch)
        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None,epoch, steps_completed=0, total_steps=1,
                                        log_freq=1, loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)

        # remember best top1 and save checkpoint
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        apputils.save_checkpoint(epoch, args.arch, model, optimizer, compression_scheduler, best_top1, is_best, args.name)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], args.print_freq)


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, print_freq, log_params_hist):
    """Training loop for one epoch."""
    losses = {'objective_loss'   : tnt.AverageValueMeter(),
              'regularizer_loss' : tnt.AverageValueMeter()}
    if compression_scheduler is None:
        # Initialize the regularizer loss to zero
        losses['regularizer_loss'].add(0)

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

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
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        # Execute the forard phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch)
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        classerr.add(output.data, target)
        losses['objective_loss'].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we add any regularization loss computed by the scheduler
            regularizer_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss)
            loss += regularizer_loss
            losses['regularizer_loss'].add(regularizer_loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % print_freq == 0:
            # Log some statistics
            lr = optimizer.param_groups[0]['lr']
            stats = ('Peformance/Training/',
                     OrderedDict([
                         ('Loss', losses['objective_loss'].mean),
                         ('Reg Loss', losses['regularizer_loss'].mean),
                         ('Top1', classerr.value(1)),
                         ('Top5', classerr.value(5)),
                         ('LR', lr),
                         ('Time', batch_time.mean)])
                    )

            distiller.log_training_progress(stats,
                                            model.named_parameters() if log_params_hist else None,
                                            epoch, steps_completed,
                                            steps_per_epoch, print_freq,
                                            loggers)
        end = time.time()


def validate(val_loader, model, criterion, loggers, print_freq, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, print_freq, epoch)


def test(test_loader, model, criterion, loggers, print_freq):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    return _validate(test_loader, model, criterion, loggers, print_freq)


def _validate(data_loader, model, criterion, loggers, print_freq, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss' : tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    # if nclasses<=10:
    #     # Log the confusion matrix only if the number of classes is small
    #     confusion = tnt.ConfusionMeter(10)

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

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
            # if confusion:
            #     confusion.add(output.data, target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % print_freq == 0:
                stats = ('',
                         OrderedDict([('Loss', losses['objective_loss'].mean),
                                      ('Top1', classerr.value(1)),
                                      ('Top5', classerr.value(5))]))
                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, print_freq, loggers)

    msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                   classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

    # if confusion:
    #     msglogger.info('==> Confusion:\n%s', str(confusion.value()))
    return classerr.value(1), classerr.value(5), losses['objective_loss'].mean


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
    if torch.__version__ >= '0.4':
        return torch.autograd.Variable(tensor)
    return torch.autograd.Variable(tensor, volatile=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if msglogger is not None:
            msglogger.error(traceback.format_exc())
        exit(1)
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))
