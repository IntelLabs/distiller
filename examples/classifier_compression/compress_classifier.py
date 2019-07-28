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
    train_one_epoch()
    validate()
    compression_scheduler.on_epoch_end(epoch)
    save_checkpoint()

training:
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import time
import os
import traceback
import logging
from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from distiller.data_loggers import *
import distiller.quantization as quantization
import examples.automated_deep_compression as adc
from distiller.models import ALL_MODEL_NAMES, create_model
from distiller import imageclass
from imageclass import *
import parser


# Logger handle
msglogger = None


def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    global msglogger

    # Parse arguments
    args = parser.get_parser().parse_args()
    if args.epochs is None:
        args.epochs = 90

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
                                         args.verbose)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    if args.evaluate:
        args.deterministic = True
    if args.deterministic:
        distiller.set_deterministic(args.seed) # For experiment reproducability
    else:
        if args.seed is not None:
            distiller.set_seed(args.seed)
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

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

    # Infer the dataset from the model name
    args.dataset = distiller.apputils.classification_dataset_str_from_arch(args.arch)
    args.num_classes = distiller.apputils.classification_num_classes(args.dataset)

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    if args.deprecated_resume:
        msglogger.warning('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            msglogger.warning('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    # We can optionally resume from a checkpoint
    optimizer = None
    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(args.device)

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    if args.AMC:
        return automated_deep_compression(model, criterion, optimizer, pylogger, args)
    if args.greedy:
        return greedy(model, criterion, optimizer, pylogger, args)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        return

    if args.export_onnx is not None:
        return distiller.export_img_classifier_to_onnx(model,
            os.path.join(msglogger.logdir, args.export_onnx),
            args.dataset, add_softmax=True, verbose=False)

    if args.qe_calibration:
        args_copy = copy.deepcopy(args)
        args_copy.effective_test_size = args.qe_calibration
        return acts_quant_stats_collection(load_data(args_copy)[2], model, criterion, pylogger,
                                           args_copy, save_to_file=True)

    if args.activation_histograms:
        return acts_histogram_collection(model, criterion, pylogger, args)

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = load_data(args)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1], args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(test_loader, model, criterion, pylogger, activations_collectors, args,
                              compression_scheduler)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) if args.resumed_checkpoint_path else None)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    if args.thinnify:
        #zeros_mask_dict = distiller.create_model_masks_dict(model)
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.remove_filters(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        print("Note: your model may have collapsed to random inference, so you may want to fine-tune")
        return

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

    if start_epoch >= ending_epoch:
        msglogger.error(
            'epoch count is too low, starting epoch is {} but total epochs set to {}'.format(
            start_epoch, ending_epoch))
        raise ValueError('Epochs parameter is too low. Nothing to do.')
    for epoch in range(start_epoch, ending_epoch):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train_one_epoch(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer, metrics={'min': vloss, 'max': top1})

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, model, top1, top5, epoch, args.num_best_scores)
        is_best = epoch == perf_scores_history[0].epoch
        checkpoint_extras = {'current_top1': top1,
                             'best_top1': perf_scores_history[0].top1,
                             'best_epoch': perf_scores_history[0].epoch}
        apputils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)


def evaluate_model(test_loader, model, criterion, loggers, activations_collectors, args, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if not args.quantize_eval:
        return test(test_loader, model, criterion, loggers, activations_collectors, args=args)
    else:
        return quantize_and_test_model(test_loader, model, criterion, loggers, activations_collectors,
                                 args=args, scheduler=scheduler, save_flag=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
