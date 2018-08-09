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

"""Policies for scheduling by a CompressionScheduler instance.

- PruningPolicy: prunning policy
- RegularizationPolicy: regulization scheduling
- LRPolicy: learning-rate decay scheduling
"""
import torch
from collections import namedtuple

import logging
msglogger = logging.getLogger()

__all__ = ['PruningPolicy', 'RegularizationPolicy', 'QuantizationPolicy', 'LRPolicy', 'ScheduledTrainingPolicy',
           'PolicyLoss', 'LossComponent']

PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):
    """ Base class for all scheduled training policies.

    The CompressionScheduler invokes these methods as the training progresses.
    """
    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_begin(self, model, zeros_mask_dict, meta):
        """A new epcoh is about to begin"""
        pass

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer=None):
        """The forward-pass of a new mini-batch is about to begin"""
        pass

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        """The mini-batch training pass has completed the forward-pass,
        and is about to begin the backward pass.

        This callback receives a 'loss' argument. The callback should not modify this argument, but it can
        optionally return an instance of 'PolicyLoss' which will be used in place of `loss'.

        Note: The 'loss_components' parameter within 'PolicyLoss' should contain any new, individual loss components
              the callback contributed to 'overall_loss'. It should not contain the incoming 'loss' argument.
        """
        pass

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        """The mini-batch training pass has ended"""
        pass

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        """The current epoch has ended"""
        pass


class PruningPolicy(ScheduledTrainingPolicy):
    """Base class for pruning policies.

    The current implementation restricts the pruning step to the beginning of
    each epoch.  This can be easily changed.
    """
    def __init__(self, pruner, pruner_args, classes=None, layers=None):
        super(PruningPolicy, self).__init__(classes, layers)
        self.pruner = pruner
        self.levels = None
        if pruner_args is not None and 'levels' in pruner_args:
            self.levels = pruner_args['levels']

    def on_epoch_begin(self, model, zeros_mask_dict, meta):
        msglogger.debug("Pruner {} is about to prune".format(self.pruner.name))
        if self.levels is not None:
            self.pruner.levels = self.levels

        for param_name, param in model.named_parameters():
            self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer=None):
        for param_name, param in model.named_parameters():
            zeros_mask_dict[param_name].apply_mask(param)


class RegularizationPolicy(ScheduledTrainingPolicy):
    """Regularization policy.

    """
    def __init__(self, regularizer, keep_mask=False):
        super(RegularizationPolicy, self).__init__()
        self.regularizer = regularizer
        self.keep_mask = keep_mask
        self.is_last_epoch = False

    def on_epoch_begin(self, model, zeros_mask_dict, meta):
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss,
                             zeros_mask_dict, optimizer=None):
        regularizer_loss = torch.tensor(0, dtype=torch.float, device=loss.device)

        for param_name, param in model.named_parameters():
            self.regularizer.loss(param, param_name, regularizer_loss, zeros_mask_dict)

        policy_loss = PolicyLoss(loss + regularizer_loss,
                                 [LossComponent(self.regularizer.__class__.__name__ + '_loss', regularizer_loss)])
        return policy_loss

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.regularizer.threshold_criteria is None:
            return

        keep_mask = False
        if (minibatches_per_epoch-1 == minibatch_id) and self.is_last_epoch and self.keep_mask:
            # If this is the last mini_batch in the last epoch, and the scheduler wants to
            # keep the regularization mask, then now is the time ;-)
            msglogger.info("RegularizationPolicy is keeping the regularization mask")
            keep_mask = True

        for param_name, param in model.named_parameters():
            self.regularizer.threshold(param, param_name, zeros_mask_dict)
            if keep_mask:
                zeros_mask_dict[param_name].is_regularization_mask = False
            zeros_mask_dict[param_name].apply_mask(param)


class LRPolicy(ScheduledTrainingPolicy):
    """ Learning-rate decay scheduling policy.

    """
    def __init__(self, lr_scheduler):
        super(LRPolicy, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_begin(self, model, zeros_mask_dict, meta):
        self.lr_scheduler.step()


class QuantizationPolicy(ScheduledTrainingPolicy):
    def __init__(self, quantizer):
        super(QuantizationPolicy, self).__init__()
        self.quantizer = quantizer
        self.quantizer.prepare_model()
        self.quantizer.quantize_params()

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        # After parameters update, quantize the parameters again
        # (Doing this here ensures the model parameters are quantized at training completion (and at validation time)
        self.quantizer.quantize_params()
