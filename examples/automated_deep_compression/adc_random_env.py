import random
import numpy as np
from scipy.stats import truncnorm
import logging

msglogger = logging.getLogger()


class RandomADCActionSpace(object):
    def __init__(self, low, high, std, mean, num_layers):
        self.action_low = low
        self.action_high = high
        self.layer = 0
        self.num_layers = num_layers
        self.means = [mean] * num_layers
        #self.means = [high-low] * self.num_layers
        #self.means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]
        self.std = std

    def sample(self):
        return random.uniform(self.action_low, self.action_high)
        # action_values_mean = self.means[self.layer]
        # action_values_std = self.std
        # normalized_low = (self.action_low - action_values_mean) / action_values_std
        # normalized_high = (self.action_high - action_values_mean) / action_values_std
        # distribution = truncnorm(normalized_low, normalized_high, loc=action_values_mean, scale=action_values_std)
        # action = distribution.rvs(1)
        # # action = np.random.normal(self.means[self.layer], self.std)
        # # action = min(self.action_high, max(action, self.action_low))
        # self.layer = (self.layer + 1) % self.num_layers
        # return action

    def set_cfg(self, means, std):
        self.means = [0.01*m for m in self.means] + [0.99*m for m in means]
        self.std = std


def random_agent(env):
    """Random ADC agent"""
    action_range = env.amc_cfg.action_range
    best_episode = [-1000, None]
    update_rate = 5
    num_layers = env.net_wrapper.num_layers()
    env.action_space = RandomADCActionSpace(action_range[0], action_range[1],
                                            std=0.35, mean=env.amc_cfg.target_density, num_layers=num_layers)
    for ep in range(1000):
        observation = env.reset()
        action_config = []
        for t in range(100):
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
            best_episode[0] = reward
            best_episode[1] = action_config
        if ep % update_rate == 0:
            env.action_space.set_cfg(means=best_episode[1], std=0.4)
            best_episode = [-1000, None]
