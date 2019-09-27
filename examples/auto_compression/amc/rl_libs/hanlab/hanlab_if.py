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

from examples.auto_compression.amc.rl_libs.hanlab.agent import DDPG, train
import logging


msglogger = logging.getLogger()


class ArgsContainer(object):
    def __init__(self):
        pass


class RlLibInterface(object):
    """Interface to a hanlab DDPG impelementation."""

    def solve(self, env, args):
        msglogger.info("AMC: Using hanlab")
        
        agent_args = ArgsContainer()
        agent_args.bsize = args.batch_size
        agent_args.tau = 0.01
        agent_args.discount = 1.
        agent_args.epsilon = 50000
        agent_args.init_delta = 0.5
        agent_args.delta_decay = 0.95
        agent_args.warmup = env.amc_cfg.ddpg_cfg.num_heatup_episodes
        agent_args.lr_c = env.amc_cfg.ddpg_cfg.critic_lr
        agent_args.lr_a = env.amc_cfg.ddpg_cfg.actor_lr
        agent_args.hidden1 = 300
        agent_args.hidden2 = 300
        agent_args.rmsize = env.amc_cfg.ddpg_cfg.replay_buffer_size
        agent_args.window_length = 1
        agent_args.train_episode = (env.amc_cfg.ddpg_cfg.num_heatup_episodes +
                                    env.amc_cfg.ddpg_cfg.num_training_episodes)
        agent_args.output = "."
        agent = DDPG(args.observation_len, 1, agent_args)
        train(agent_args.train_episode, agent, env, agent_args.output, agent_args.warmup)
