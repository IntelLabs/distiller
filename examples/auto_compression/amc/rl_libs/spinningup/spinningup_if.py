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

    def solve(self, env1, env2):
        msglogger.info("AMC: Using Spinningup")

        # training_noise_duration = amc_cfg.num_training_episodes * steps_per_episode
        heatup_duration = env1.amc_cfg.ddpg_cfg.num_heatup_episodes * env1.steps_per_episode

        exp_name = "Test"
        seed = 0
        # The number and size of the Actor-Critic MLP hidden layers
        layers, hid = 2, 300
        logger_kwargs = setup_logger_kwargs(exp_name)

        ddpg.ddpg(env=env1, test_env=env2, actor_critic=core.mlp_actor_critic,
                  ac_kwargs=dict(hidden_sizes=[hid]*layers, output_activation=tf.sigmoid),
                  gamma=1,  # discount rate
                  #seed=seed,
                  epochs=400,
                  replay_size=env1.amc_cfg.ddpg_cfg.replay_buffer_size,
                  batch_size=64,
                  start_steps=heatup_duration,
                  steps_per_epoch=env1.steps_per_episode,
                  act_noise=env1.amc_cfg.ddpg_cfg.initial_training_noise,
                  pi_lr=env1.amc_cfg.ddpg_cfg.actor_lr,
                  q_lr=env1.amc_cfg.ddpg_cfg.critic_lr,
                  logger_kwargs=logger_kwargs,
                  noise_decay=env1.amc_cfg.ddpg_cfg.training_noise_decay)

