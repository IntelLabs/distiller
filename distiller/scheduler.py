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
from functools import partial
import logging
import torch
from .quantization.quantizer import FP_BKP_PREFIX
from .policy import PolicyLoss, LossComponent

msglogger = logging.getLogger()


class ParameterMasker(object):
    def __init__(self, param_name):
        msglogger.debug('Created masker for parameter {0}'.format(param_name))
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.is_regularization_mask = False

    def apply_mask(self, tensor, in_backward_cb=False):
        """Apply a mask on the tensor.

        The tensor is either a gradients tensor (when apply_mask is invoked from the
        backward hook of the variable owning the gradient); or a weights tensor
        (when apply_mask is invoked by the scheduler).
        """
        if self.mask is None:
            msglogger.debug('No mask for parameter {0}'.format(self.param_name))
            return
        if in_backward_cb and self.is_regularization_mask:
            # We don't want to mask gradients when a regularizer generated the mask.
            return
        msglogger.debug('Masking parameter {0}'.format(self.param_name))
        tensor.data.mul_(self.mask)
        if self.is_regularization_mask:
            self.mask = None
        return tensor


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
            param.register_hook(partial(masker.apply_mask, in_backward_cb=True))

    def add_policy(self, policy, epochs=None, starting_epoch=None, ending_epoch=None, frequency=None):
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

    def on_epoch_begin(self, epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta)

    def on_minibatch_begin(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy.on_minibatch_begin(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                          self.zeros_mask_dict, optimizer)

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
        # When we get to this point, the weights are no longer maksed.  This is because during the backward
        # pass, the weights are updated.  So we choose to lazily apply the pruning mask, only if some
        # component is being called-back.
        weights_are_masked = False

        if epoch in self.policies:
            for policy in self.policies[epoch]:
                if not weights_are_masked:
                    self.apply_mask()
                    weights_are_masked = True
                policy.on_minibatch_end(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                        self.zeros_mask_dict, optimizer)

    def on_epoch_end(self, epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                meta['optimizer'] = optimizer
                policy.on_epoch_end(self.model, self.zeros_mask_dict, meta)

    def apply_mask(self):
        for name, param in self.model.named_parameters():
            try:
                self.zeros_mask_dict[name].apply_mask(param)
            except KeyError:
                # Quantizers for training modify some model parameters by adding a prefix
                # If this is the source of the error, workaround and move on
                name_parts = name.split('.')
                if name_parts[-1].startswith(FP_BKP_PREFIX):
                    name_parts[-1] = name_parts[-1].replace(FP_BKP_PREFIX, '', 1)
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

    def load_state_dict(self, state):
        """Loads the scheduler state.

        Currently the scheduler state is comprised only of the set of pruning masks.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.  It is a dictionary of parameter
                names (keys) and parameter masks (values).
        """
        try:
            loaded_masks = state['masks_dict']
        except Exception as exception:
            print("ERROR: could not load the CompressionScheduler state")
            print("Exception: %s %s" % (type(exception), exception))
            print("\t\tFound the following keys in the state dictionary:")
            for k in state.keys():
                print("\t\t" + k)
            exit(1)

        for name, mask in self.zeros_mask_dict.items():
            masker = self.zeros_mask_dict[name]
            masker.mask = loaded_masks[name]

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
