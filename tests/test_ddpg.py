from ddpg import Ddpg
from gym import spaces
import numpy as np
import math
import tensorflow as tf

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
    agent.compute_actions = lambda x, **kwargs: np.zeros_like(x)

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
    for i in range(500):
        agent.train_critic(sars)

    predict_1 = agent.critic(observations=np.zeros([1, 1]), actions=np.zeros([1, 1]))
    assert math.isclose(predict_1.numpy()[0,0], 1, abs_tol=2e-2)


def test_gradient():

    x = tf.Variable([1])
    with tf.GradientTape() as tape:
        y = x

    assert tape.gradient(y, x).numpy()[0] == 1

def test_optimizer():
    opt = tf.optimizers.Adam()
    x = tf.Variable([1], dtype=tf.float32)
    dx = tf.ones([1], dtype=tf.float32)

    opt.apply_gradients( [(dx, x)] )
    # surprize - tf actually follows negative of the provided gradient.
    assert x.numpy()[0] < 1


def test_compute_actions_range_high():
    desired_range = spaces.Box(50 * np.ones([1]), 100 * np.ones([1]), dtype=np.float32)

    agent = Ddpg(action_space=desired_range,
                 obs_space=two_unit_space,
                 buffer_size=100)

    observation = np.random.rand(5,1)

    actions = agent.compute_actions(observation).numpy()

    assert actions.shape == (5,1)
    right_range = np.logical_and(50 <= actions, actions <= 100)
    assert right_range.all()


def test_compute_actions_range_high_tf():
    desired_range = spaces.Box(50 * np.ones([1]), 100 * np.ones([1]), dtype=np.float32)

    agent = Ddpg(action_space=desired_range,
                 obs_space=two_unit_space,
                 buffer_size=100)

    observation = np.random.rand(5,1)

    actions = agent.compute_actions(observation)

    assert actions.shape == (5,1)
    assert tf.reduce_all( tf.logical_and(  50 <= actions,  actions <= 100  ) )


def test_env_action():
    desired_range = spaces.Box(-5 * np.ones([1]), 10 * np.ones([1]), dtype=np.float32)

    agent = Ddpg(action_space=desired_range,
                 obs_space=two_unit_space,
                 buffer_size=100,
                 epsilon=0)

    observation = np.random.rand(1)

    actions = agent.env_action(observation)

    assert actions.shape == (1,)
    assert tf.reduce_all(tf.logical_and(-5 <= actions, actions <= 10))

def test_env_random_action():
    desired_range = spaces.Box(-5 * np.ones([1]), 10 * np.ones([1]), dtype=np.float32)

    agent = Ddpg(action_space=desired_range,
                 obs_space=two_unit_space,
                 buffer_size=100,
                 epsilon=1)

    observation = np.random.rand(1)

    actions = agent.env_action(observation)

    assert actions.shape == (1,)
    assert tf.reduce_all(tf.logical_and(-5 <= actions, actions <= 10))













