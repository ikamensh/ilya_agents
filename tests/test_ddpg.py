from ddpg import Ddpg
from gym import spaces
import numpy as np
import math

two_unit_space = spaces.Box(-np.ones([1]), np.ones([1]), dtype=np.float32)

def test_constructor():
    agent = Ddpg(action_space=two_unit_space,
                 obs_space=two_unit_space,
                 buffer_size=100)


def test_critic():
    """ Test critic can learn:
    state 0 leads to state 1 with +1 reward,
    state 1 leads to state 1 with 0 reward.
    """
    agent = Ddpg(action_space=two_unit_space,
                 obs_space=two_unit_space,
                 buffer_size=100,
                 critic_lr=0.02)
    agent.act = lambda x: np.zeros_like(x)

    for i in range(10):
        agent.add_xp(obs=np.zeros([1]),
                     action=np.zeros([1]),
                     reward=1,
                     obs_new=np.ones([1]))

        agent.add_xp(obs=np.ones([1]),
                     action=np.zeros([1]),
                     reward=0,
                     obs_new=np.ones([1]))

    sars = agent.buffer.get_batch(100)
    for i in range(1000):
        agent.train_critic(sars)

    predict_1 = agent.critic(observations=np.zeros([1, 1]), actions=np.zeros([1, 1]))
    assert math.isclose(predict_1.numpy()[0,0], 1)




