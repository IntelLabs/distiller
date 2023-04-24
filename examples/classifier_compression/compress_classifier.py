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
    compression_scheduler.on_epoch_end(epoch)
    save_checkpoint()

train():
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

import traceback
import logging
from functools import partial
import distiller
from distiller.models import create_model
import distiller.apputils.image_classifier as classifier
import distiller.apputils as apputils
import parser
import os
import numpy as np
from ptq_lapq import image_classifier_ptq_lapq


# Logger handle
msglogger = logging.getLogger()


def main():
    # Parse arguments
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    app = ClassifierCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
    if app.handle_subapps():
        return
    init_knowledge_distillation(app.args, app.model, app.compression_scheduler)
    app.run_training_loop()
    # Finally run results on the test set
    return app.test()

    
def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader

    do_exit = False
    if args.greedy:
        greedy(model, criterion, optimizer, pylogger, args)
        do_exit = True
    elif args.summary:
        # This sample application can be invoked to produce various summary reports
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        do_exit = True
    elif args.export_onnx is not None:
        distiller.export_img_classifier_to_onnx(model,
                                                os.path.join(msglogger.logdir, args.export_onnx),
                                                args.dataset, add_softmax=True, verbose=False)
        do_exit = True
    elif args.qe_calibration and not (args.evaluate and args.quantize_eval):
        classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.activation_histograms:
        classifier.acts_histogram_collection(model, criterion, pylogger, args)
        do_exit = True
    elif args.sensitivity is not None:
        test_loader = load_test_data(args)
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        if args.quantize_eval and args.qe_lapq:
            image_classifier_ptq_lapq(model, criterion, pylogger, args)
        else:
            test_loader = load_test_data(args)
            classifier.evaluate_model(test_loader, model, criterion, pylogger,
                classifier.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=compression_scheduler)
        do_exit = True
    elif args.thinnify:
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.contract_model(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        msglogger.info("Note: if your model collapsed to random inference, you may want to fine-tune")
        do_exit = True
    return do_exit


def init_knowledge_distillation(args, model, compression_scheduler):
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


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class ClassifierCompressorSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)


    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(classifier.test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=classifier.create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.png'))
    distiller.sensitivities_to_csv(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.csv'))


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader = classifier.load_data(args)

    test_fn = partial(classifier.test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(classifier.train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


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
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
