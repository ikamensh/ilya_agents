import gym
import numpy as np
import copy

class Integrating(gym.core.Env):
    action_space = gym.spaces.Box(-np.ones([1]), np.ones([1]))
    observation_space = gym.spaces.Box(np.zeros([1]), np.ones([1]))
    max_turns = 100


    def render(self, mode='human'):
        pass

    def __init__(self):
        self.done = False
        self.ctr = 0
        self.state = np.zeros([1])

    def reset(self):
        self.done = False
        self.ctr = 0
        self.state = np.zeros([1])
        return copy.copy( self.state )

    def step(self, action):
        if self.done:
            raise Exception("reset the env!")

        self.ctr += 1
        if self.ctr > self.max_turns:
            self.done = True
        return copy.copy( self.state ), np.squeeze(action), self.done, {}

    def __str__(self):
        return "Integrating environment"


if __name__ == "__main__":
    env = Integrating()

    env.reset()
    print(env.step(0))
    print(env.step(1))