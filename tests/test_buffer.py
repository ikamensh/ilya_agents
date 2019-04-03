from replay_buffer import ReplayBuffer

import numpy as np


def test_valid_batch():
    buf = ReplayBuffer(100)

    for i in range(70):
        buf.add_xp(obs=np.ones([1]),
                   action=np.ones([1]),
                   reward=1,
                   obs_new=np.ones([1]))

    obs1, actions, rewards, obs2 = buf.get_batch(32)

    assert hasattr(obs1, 'shape')
    assert len(obs1.shape) == 2
    assert obs1.shape[0] == 32

    assert hasattr(actions, 'shape')
    assert len(actions.shape) == 2
    assert obs1.shape[0] == 32

    assert hasattr(rewards, 'shape')
    assert len(rewards.shape) == 2
    assert obs1.shape[0] == 32

    assert hasattr(obs2, 'shape')
    assert len(obs2.shape) == 2
    assert obs1.shape[0] == 32

def test_randomized():
    buf = ReplayBuffer(100)

    for i in range(70):
        buf.add_xp(obs=np.random.rand(1),
                   action=np.random.rand(1),
                   reward=1,
                   obs_new=np.random.rand(1))

    batch1 = buf.get_batch(32)
    batch2 = buf.get_batch(32)

    assert not np.all(np.vstack(batch1) == np.vstack(batch2))

