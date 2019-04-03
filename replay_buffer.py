from collections import deque, namedtuple
import random
import numpy as np

experience = namedtuple('xp', 'obs1 action reward obs2')


class ReplayBuffer:

    def __init__(self, maxlen):
        self.deque = deque(maxlen=maxlen)

    def add_xp(self, obs, action, reward, obs_new):
        self.deque.append( experience(obs, action, reward, obs_new) )


    def get_batch(self, size: int):

        size = min(len(self.deque), size)

        batch = random.sample(self.deque, size)

        obs1 = np.vstack( [xp.obs1 for xp in batch] )
        actions = np.vstack( [xp.action for xp in batch] )
        rewards = np.vstack( [xp.reward for xp in batch] )
        obs2 = np.vstack( [xp.obs2 for xp in batch] )

        for i, tensor in enumerate([obs1, actions, rewards, obs2]):
            assert hasattr(tensor, 'shape'), f"{i}:{tensor} must be a np / tf tensor"
            assert len(tensor.shape) is 2, f"{i}: must be at most 2d, found: {tensor.shape}"
            assert tensor.shape[0] == size, \
                f"batch must have first dim == batchsize, found {tensor.shape[0]}"

        return obs1, actions, rewards, obs2


