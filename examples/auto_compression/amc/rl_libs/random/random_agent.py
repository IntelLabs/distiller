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

import random
import numpy as np
from scipy.stats import truncnorm
import logging

msglogger = logging.getLogger()


class RandomADCActionSpace(object):
    def __init__(self, low, high):
        self.action_low = low
        self.action_high = high

    def sample(self):
        return random.uniform(self.action_low, self.action_high)


def random_agent(env):
    """Random ADC agent"""
    action_range = env.amc_cfg.action_range
    best_episode = [float("-inf"), None]
    env.action_space = RandomADCActionSpace(action_range[0], action_range[1])
    n_episodes = env.amc_cfg.ddpg_cfg.num_training_episodes + env.amc_cfg.ddpg_cfg.num_heatup_episodes 
    for ep in range(n_episodes):
        observation = env.reset()
        action_config = []
        for t in range(env.steps_per_episode):
            #env.render(0, 0)
            msglogger.info("[episode={}:{}] observation = {}".format(ep, t, observation))
            # take a random action
            action = env.action_space.sample()
            action_config.append(action)
            observation, reward, done, info = env.step([action])
            if done:
                msglogger.info("Episode finished after {} timesteps".format(t+1))
                msglogger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                msglogger.info("New solution found: episode={} reward={} config={}".format(ep, reward, action_config))
                break
        if reward > best_episode[0]:
            best_episode = (reward, action_config)