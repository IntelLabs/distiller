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

"""Compression scheduling.

This implements the scheduling of the compression policies.
"""
import contextlib
from functools import partial
import logging

import torch
from .quantization.quantizer import FP_BKP_PREFIX
from .policy import PolicyLoss, LossComponent
from .utils import model_device, normalize_module_name
msglogger = logging.getLogger()


class ParameterMasker(object):
    def __init__(self, param_name):
        msglogger.debug('Created masker for parameter {0}'.format(param_name))
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None

    def apply_mask(self, tensor):
        """Apply a mask on the weights tensor."""
        if self.mask is None:
            msglogger.debug('No mask for parameter {0}'.format(self.param_name))
            return
        msglogger.debug('Masking parameter {0}'.format(self.param_name))
        if self.use_double_copies:
            self.unmasked_copy = tensor.clone()
        tensor.data.mul_(self.mask)
        if self.is_regularization_mask:
            self.mask = None
        return tensor

    def remove_mask(self, tensor):
        if self.mask is None:
            msglogger.debug('No mask for parameter {0}'.format(self.param_name))
            return
        if not self.use_double_copies:
            msglogger.debug('Parameter {0} does not maintain double copies'.format(self.param_name))
            return
        tensor.data = self.unmasked_copy.data


def create_model_masks_dict(model):
    """A convinience function to create a dictionary of paramter maskers for a model"""
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return zeros_mask_dict


class CompressionScheduler(object):
    """Responsible for scheduling pruning and masking parameters.

    """
    def __init__(self, model, device=torch.device("cuda")):
        self.model = model
        self.device = device
        self.policies = {}
        self.sched_metadata = {}
        self.zeros_mask_dict = {}
        for name, param in self.model.named_parameters():
            masker = ParameterMasker(name)
            self.zeros_mask_dict[name] = masker

    def add_policy(self, policy, epochs=None, starting_epoch=0, ending_epoch=1, frequency=1):
        """Add a new policy to the schedule.

        Args:
            epochs (list): A list, or range, of epochs in which to apply the policy
        """

        if epochs is None:
            epochs = list(range(starting_epoch, ending_epoch, frequency))

        for epoch in epochs:
            if epoch not in self.policies:
                self.policies[epoch] = [policy]
            else:
                self.policies[epoch].append(policy)
            assert len(self.policies[epoch]) > 0

        self.sched_metadata[policy] = {'starting_epoch': starting_epoch,
                                       'ending_epoch': ending_epoch,
                                       'frequency': frequency}

    def on_epoch_begin(self, epoch, optimizer=None, **kwargs):
        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta,
                                  **kwargs)

    def on_minibatch_begin(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                policy.on_minibatch_begin(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                          self.zeros_mask_dict, meta, optimizer)

    def before_backward_pass(self, epoch, minibatch_id, minibatches_per_epoch, loss, optimizer=None,
                             return_loss_components=False):
        # We pass the loss to the policies, which may override it
        overall_loss = loss
        loss_components = []
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy_loss = policy.before_backward_pass(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                          overall_loss, self.zeros_mask_dict)
                if policy_loss is not None:
                    curr_loss_components = self.verify_policy_loss(policy_loss)
                    overall_loss = policy_loss.overall_loss
                    loss_components += curr_loss_components

        if return_loss_components:
            return PolicyLoss(overall_loss, loss_components)

        return overall_loss

    def on_minibatch_end(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        # When we get to this point, the weights are no longer masked.  This is because during the backward
        # pass, the weights may have been updated.  This is true even when the gradients are zero, for some
        # optimization algorithms such as SGD with momentum.  See the Note in PyTorch's SGD documentation:
        # https://pytorch.org/docs/stable/optim.html#torch.optim.SGD.
        #
        # Therefore we choose to always apply the pruning mask.  In the future we may optimize this by applying
        # the mask only if the some policy is actually using the mask.
        self.apply_mask(is_forward=False)
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy.on_minibatch_end(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                        self.zeros_mask_dict, optimizer)

    def on_epoch_end(self, epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                meta['optimizer'] = optimizer
                policy.on_epoch_end(self.model, self.zeros_mask_dict, meta)

    def apply_mask(self, is_forward=True):
        for name, param in self.model.named_parameters():
            try:
                if is_forward or not self.zeros_mask_dict[name].mask_on_forward_only:
                    # When we mask on forward-pass only, we allow the gradients to change
                    # the weights.
                    self.zeros_mask_dict[name].apply_mask(param)
            except KeyError:
                # Quantizers for training might modify model parameters in a couple of ways:
                #   1. By adding a prefix to the parameter tensor name
                #   2. By wrapping the module holding the parameter in a wrapper module
                # If the source of the error is one of the above, workaround and move on
                #
                # Quantizers might also add new learnable parameters (e.g. the clip value in PACT quantization)
                # These parameters will also be missing from the masks mapping. For now, we'll assume that we're
                # not interested in pruning these parameters - and we just ignore them.
                #
                # TODO: This is not scalable at all. Find a solution that doesn't "hard-code" these conditions...
                name_parts = name.split('.')
                prefixed = name_parts[-1].startswith(FP_BKP_PREFIX)
                wrapped = name_parts[-2] == 'wrapped_module'
                if prefixed or wrapped:
                    if prefixed:
                        name_parts[-1] = name_parts[-1].replace(FP_BKP_PREFIX, , 1)
                    if wrapped:
                        name_parts.pop(-2)
                    name = '.'.join(name_parts)
                    self.zeros_mask_dict[name].apply_mask(param)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        Currently it contains just the pruning mask.
        """
        masks = {}
        for name, masker in self.zeros_mask_dict.items():
            masks[name] = masker.mask
        state = {'masks_dict': masks}
        return state

    def load_state_dict(self, state, normalize_dataparallel_keys=False):
        """Loads the scheduler state.

        Currently the scheduler state is comprised only of the set of pruning masks.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`. It is a dictionary of parameter
                names (keys) and parameter masks (values).
            normalize_dataparallel_keys (bool): indicates if we should convert the keys from
                DataParallel format.  This should be set to True when loading a model
                from a GPU-checkpoint onto a CPU (because currently we don't use DataParallel
                on the CPU).
        """
        try:
            loaded_masks = state['masks_dict']
        except KeyError as exception:
            msglogger.error('could not load the CompressionScheduler state.'
                ' masks_dict is missing from state')
            with contextlib.suppress(TypeError):
                msglogger.debug('Scheduler state keys are: {}'.format(', '.join(state)))
            raise

        if normalize_dataparallel_keys:
            loaded_masks = {normalize_module_name(k): v for k, v in loaded_masks.items()}
        device = model_device(self.model)
        for name, mask in self.zeros_mask_dict.items():
            masker = self.zeros_mask_dict[name]
            masker.mask = loaded_masks[name]
            if masker.mask is not None:
                masker.mask = masker.mask.to(device)

    @staticmethod
    def verify_policy_loss(policy_loss):
        if not isinstance(policy_loss, PolicyLoss):
            raise TypeError("A Policy's before_backward_pass must return either None or an instance of " +
                            PolicyLoss.__name__)
        curr_loss_components = policy_loss.loss_components
        if not isinstance(curr_loss_components, list):
            curr_loss_components = [curr_loss_components]
        if not all(isinstance(lc, LossComponent) for lc in curr_loss_components):
            raise TypeError("Expected an instance of " + LossComponent.__name__ +
                            " or a list of such instances")
        return curr_loss_components
