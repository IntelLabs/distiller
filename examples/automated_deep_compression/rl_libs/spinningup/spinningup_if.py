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

import logging
import tensorflow as tf
from spinup.algos.ddpg import core
from spinup.utils.run_utils import setup_logger_kwargs
from .ddpg import ddpg


msglogger = logging.getLogger()


class RlLibInterface(object):
    """Interface to the Spinningup RL library"""

    def solve(self, env1, env2, num_layers):
        msglogger.info("AMC: Using Spinningup")

        # training_noise_duration = amc_cfg.num_training_epochs * steps_per_episode
        # heatup_duration = amc_cfg.num_heatup_epochs * steps_per_episode

        exp_name = "Test"
        seed = 0
        # The number and size of the Actor-Critic MLP hidden layers
        layers, hid = 2, 300
        logger_kwargs = setup_logger_kwargs(exp_name)  # ,  seed)

        ddpg.ddpg(env=env1, test_env=env2, actor_critic=core.mlp_actor_critic,
                  ac_kwargs=dict(hidden_sizes=[hid]*layers, output_activation=tf.sigmoid),
                  gamma=1,  # discount rate
                  seed=seed,
                  epochs=400,
                  replay_size=2000,
                  batch_size=64,
                  start_steps=env1.amc_cfg.num_heatup_epochs,
                  steps_per_epoch=800 * num_layers,  # every 50 episodes perform 10 episodes of testing
                  act_noise=0.5,
                  pi_lr=1e-4,
                  q_lr=1e-3,
                  logger_kwargs=logger_kwargs)

