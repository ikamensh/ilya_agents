import gym
import numpy as np

class Ez(gym.core.Env):
    action_space = gym.spaces.Box(-np.ones([1]), np.ones([1]))
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

    def reset(self):
        self.done = False
        self.ctr = 0
        return self.useless_obs

    def step(self, action):
        if self.done:
            raise Exception("reset the env!")

        self.ctr += 1
        if self.ctr > self.max_turns:
            self.done = True
        return self.useless_obs, np.squeeze(action), self.done, {}

    def __str__(self):
        return "Linear environment"


if __name__ == "__main__":
    env = Ez()

    env.reset()
    print(env.step(0))
    print(env.step(1))