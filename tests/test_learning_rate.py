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

import pytest
import torch
from torch.optim import Optimizer
from distiller.learning_rate import MultiStepMultiGammaLR


@pytest.mark.filterwarnings('ignore:Detected call of')
def test_multi_step_multi_gamma_lr():
    dummy_tensor = torch.zeros(3, 3, 3, requires_grad=True)
    dummy_optimizer = Optimizer([dummy_tensor], {'lr': 0.1})

    # Test input checks
    with pytest.raises(ValueError):
        lr_sched = MultiStepMultiGammaLR(dummy_optimizer, milestones=[60, 30, 80], gammas=[0.1, 0.1, 0.2])
    with pytest.raises(ValueError):
        lr_sched = MultiStepMultiGammaLR(dummy_optimizer, milestones=[30, 60], gammas=[0.1, 0.1, 0.2])
    with pytest.raises(ValueError):
        lr_sched = MultiStepMultiGammaLR(dummy_optimizer, milestones=[30, 60, 80], gammas=[0.1, 0.1])

    # Test functionality
    lr_sched = MultiStepMultiGammaLR(dummy_optimizer, milestones=[30, 60, 80], gammas=[0.1, 0.1, 0.2])
    expected_gammas = [1, 1 * 0.1, 1 * 0.1 * 0.1, 1 * 0.1 * 0.1 * 0.2]
    expected_lrs = [0.1 * gamma for gamma in expected_gammas]
    assert lr_sched.multiplicative_gammas == expected_gammas
    lr_sched.step(0)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[0]
    lr_sched.step(15)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[0]
    lr_sched.step(30)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[1]
    lr_sched.step(33)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[1]
    lr_sched.step(60)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[2]
    lr_sched.step(79)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[2]
    lr_sched.step(80)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[3]
    lr_sched.step(100)
    assert dummy_optimizer.param_groups[0]['lr'] == expected_lrs[3]
