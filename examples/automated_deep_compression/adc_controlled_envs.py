"""This file contains a couple of environments used for debugging ADC reproduction.
"""
import random
import numpy as np
from scipy.stats import truncnorm


class RandomADCActionSpace(object):
    def __init__(self, low, high, std):
        self.clip_low = low
        self.clip_high = high
        self.layer = 0
        self.num_layers = 13
        #self.means = [high-low] * self.num_layers
        self.means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]
        self.std = std

    def sample(self):
        return random.uniform(self.clip_low, self.clip_high)
        action_values_mean = self.means[self.layer]
        action_values_std = self.std
        normalized_low = (self.clip_low - action_values_mean) / action_values_std
        normalized_high = (self.clip_high - action_values_mean) / action_values_std
        distribution = truncnorm(normalized_low, normalized_high, loc=action_values_mean, scale=action_values_std)
        action = distribution.rvs(1)
        # action = np.random.normal(self.means[self.layer], self.std)
        # action = min(self.clip_high, max(action, self.clip_low))
        self.layer = (self.layer + 1) % self.num_layers
        return action

    def set_cfg(self, means, std):
        self.means = [0.01*m for m in self.means] + [0.99*m for m in means]
        self.std = std


class PredictableADCActionSpace(object):
    def __init__(self, low, high):
        #self.actions = (0.51, 0.26, 0.23, 0.09, 0.24, 0.36, 0.90, 0.97, 0.98, 0.98, 0.98, 0.98, 0)
        #self.actions = (0.51, 0.26, 0.23, 0.09, 0.24, 0.36, 0.0, 0.0, 0.50, 0.50, 0.50, 0.50, 0)
        #self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.65, 0.60, 0.00, 0.00, 0)
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.00, 0)   # Top1 90.100000    Top5 99.420000    reward -0.113175
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.7, 0.00, 0)  # Top1 90.540000    Top5 99.360000    reward -0.124923
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.6, 0.00, 0)  # Top1 90.600000    Top5 99.340000    reward -0.128869

        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.8, 0)   # Top1 87.600000    Top5 98.980000    reward -0.198718
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.8, 0.65)  # Top1 74.720000    Top5 97.700000    reward -0.447991
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.8, 0.8, 0.8, 0.65) # Top1 39.540000    Top5 95.420000    reward -0.886748

        #self.actions = [0] * 13                                                           # Top1 90.480000    Top5 99.400000    reward -0.117374
        self.step = 0
        self.episode = 0
        self.l1 = 0
        self.l2 = 0
        self.update_action_vector()

    def update_action_vector(self):
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, 0.8, 0.05, 0)  # Top1 89.640000    Top5 99.520000    reward -0.093653
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, 0.8, 0.05, self.episode * 0.05)
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, self.l1 * 0.05, self.l2 * 0.05, 0)

    def sample(self):
        action = self.actions[self.step]
        self.step = (self.step + 1) % len(self.actions)
        if self.step == 0:
            self.l1 = (self.l1 + 1) % 20
            if self.l1 == 0:
                self.l2 = (self.l2 + 1) % 20
            if self.l2 == 19:
                print("Done - exiting")
                exit()
            self.update_action_vector()
        return action
