import gym
import numpy as np
import random

class Exponential(gym.core.Env):
    action_space = gym.spaces.Box(-10 * np.ones([1]), np.zeros([1]))

    observation_space = gym.spaces.Box(np.zeros([1]), np.ones([1]))
    max_turns = 100


    @property
    def useless_obs(self):
        return np.random.rand(1, 1)

    def render(self, mode='human'):
        pass

    def __init__(self):
        self.done = False
        self.ctr = 0
        self.optimal_log = -4 * random.random() - 2

    def reset(self):
        self.done = False
        self.ctr = 0
        return self.useless_obs

    def step(self, action):
        if self.done:
            raise Exception("reset the env!")

        self.ctr += 1
        if self.ctr >= self.max_turns:
            self.done = True

        action_float =  np.squeeze(action)
        log_miss = abs( self.optimal_log - action_float )
        reward = 10 ** (-log_miss/ 10)

        return self.useless_obs, reward, self.done, {}

