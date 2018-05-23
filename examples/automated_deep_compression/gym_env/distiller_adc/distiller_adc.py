import numpy as np
import gym
from gym import spaces
from enum import Enum


class AutomatedDeepCompression(gym.Env):
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30
    # }

    class ObservationType(Enum):
        OneHot = 0
        Therm = 1

    def __init__(self, chain_length=16, start_state=1, max_steps=None, observation_type=ObservationType.Therm,
                 left_state_reward=1/1000, right_state_reward=1, simple_render=True):
        super().__init__()

        print("Loaded 2")
        if chain_length <= 3:
            raise ValueError('Chain length must be > 3, found {}'.format(chain_length))
        if not 0 <= start_state < chain_length:
            raise ValueError('The start state should be within the chain bounds, found {}'.format(start_state))
        self.chain_length = chain_length
        self.start_state = start_state
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.left_state_reward = left_state_reward
        self.right_state_reward = right_state_reward
        self.simple_render = simple_render

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(0, 1, shape=(1,)) # spaces.Discrete(2)  # 0 -> Go left, 1 -> Go right
        self.observation_space = spaces.Box(0, 1, shape=(chain_length,))#spaces.MultiBinary(chain_length)

        self.reset()

    def _terminate(self):
        # Neta
        self.max_steps = 100
        return self.steps >= self.max_steps

    def _reward(self):
        if self.state == 0:
            return self.left_state_reward
        elif self.state == self.chain_length - 1:
            return self.right_state_reward
        else:
            return 0

    def _step(self, action):
        # action is 0 or 1
        # if action == 0:
        #     if 0 < self.state:
        #         self.state -= 1
        # elif action == 1:
        #     if self.state < self.chain_length - 1:
        #         self.state += 1
        # else:
        #     raise ValueError("An invalid action was given. The available actions are - 0 or 1, found {}".format(action))

        self.steps += 1

        return self._get_obs(), self._reward(), self._terminate(), {}

    def _reset(self):
        self.steps = 0

        self.state = self.start_state

        return self._get_obs()

    def _get_obs(self):
        self.observation = np.zeros((self.chain_length,))
        if self.observation_type == self.ObservationType.OneHot:
            self.observation[self.state] = 1
        elif self.observation_type == self.ObservationType.Therm:
            self.observation[:(self.state+1)] = 1

        return self.observation

    def _render(self, mode, close):
        print("render(): mode={}  close={}".format(mode, close))
        return np.zeros(1)
