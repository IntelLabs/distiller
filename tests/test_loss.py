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

import torch
import torch.nn as nn
from copy import deepcopy
import pytest

from distiller import ScheduledTrainingPolicy, CompressionScheduler
from distiller.policy import PolicyLoss, LossComponent


class DummyPolicy(ScheduledTrainingPolicy):
    def __init__(self, idx):
        super(DummyPolicy, self).__init__()
        self.loss_val = torch.randint(0, 10000, (1,))
        self.idx = idx

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss,
                             zeros_mask_dict, optimizer=None):
        return PolicyLoss(loss + self.loss_val, [LossComponent('Dummy Loss ' + str(self.idx), self.loss_val)])


@pytest.mark.parametrize("check_loss_components", [False, True])
def test_multiple_policies_loss(check_loss_components):
    model = nn.Module()
    scheduler = CompressionScheduler(model, device=torch.device('cpu'))
    num_policies = 3
    expected_overall_loss = 0
    expected_policy_losses = []
    for i in range(num_policies):
        policy = DummyPolicy(i)
        expected_overall_loss += policy.loss_val
        expected_policy_losses.append(policy.loss_val)
        scheduler.add_policy(policy, epochs=[0])

    main_loss = torch.randint(0, 10000, (1,))
    expected_overall_loss += main_loss
    main_loss_before = deepcopy(main_loss)

    policies_loss = scheduler.before_backward_pass(0, 0, 1, main_loss, return_loss_components=check_loss_components)

    assert main_loss_before == main_loss
    if check_loss_components:
        assert expected_overall_loss == policies_loss.overall_loss
        for idx, lc in enumerate(policies_loss.loss_components):
            assert lc.name == 'Dummy Loss ' + str(idx)
            assert expected_policy_losses[idx] == lc.value.item()
    else:
        assert expected_overall_loss == policies_loss
