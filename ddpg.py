import random
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, Sequential
from gym import spaces

from replay_buffer import ReplayBuffer
from utils import clone_with_weights, track_model

class Ddpg:

    def __init__(self, action_space: spaces.Box, obs_space: spaces.Space,
                 buffer_size, epsilon = 0.1, critic_lr = None, actor_lr = None):

        assert len(action_space.shape) == 1
        assert len(obs_space.shape) == 1
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

        self.scale = action_space.high - action_space.low
        self.offset = action_space.low
        self.action_space = action_space

        n_actions = action_space.shape[0]
        n_inputs = obs_space.shape[0]
        critic = Sequential()
        critic.add(layers.Dense(32, input_shape=[n_inputs + n_actions], activation='relu'))
        critic.add(layers.Dense(32, activation='relu'))
        critic.add(layers.Dense(1))
        critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr or 1e-3)
        critic.compile(optimizer=critic_optimizer, loss='mse')
        self.critic_net = critic
        self.critic_t = clone_with_weights(critic)

        actor = Sequential()
        actor.add(layers.Dense(32, input_shape=[n_inputs], activation='relu'))
        actor.add(layers.Dense(32, activation='relu'))
        actor.add(layers.Dense(n_actions, activation='sigmoid'))
        self.actor = actor
        self.actor_t = clone_with_weights(actor)

        self.optimizer = tf.optimizers.Adam(lr = actor_lr or 1e-3)

        self.buffer = ReplayBuffer(maxlen=buffer_size)

        self.future_discount = 0.99
        self.tracking_speed = 0.01

    @contextmanager
    def no_exploration(self):
        epsilon = self.epsilon
        self.epsilon = 0
        yield
        self.epsilon = epsilon

    def compute_actions(self, observation, use_target_network = False):

        assert hasattr(observation, 'shape'), f"{observation} must be a np / tf tensor"
        assert len(observation.shape) is 2, f"must be 2d, found: {observation.shape}"

        network = self.actor_t if use_target_network else self.actor

        return self.scale * network(observation) + self.offset

    def env_action(self, observation):
        assert hasattr(observation, 'shape'), f"{observation} must be a np / tf tensor"
        assert len(observation.shape) in [1,2], f"must be at most 2d, found: {observation.shape}"

        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0)

        if self.epsilon > random.random():
            return self.action_space.sample()
        else:
            out = self.compute_actions(observation)
            return tf.squeeze(out, [0])


    def learn(self, *, batch_size):
        sars = self.buffer.get_batch(batch_size)
        self.train_critic(sars)
        self.train_actor(sars)
        track_model(model=self.critic_net, tracker=self.critic_t, lr=self.tracking_speed)
        track_model(model=self.actor, tracker=self.actor_t, lr=self.tracking_speed)

    def add_xp(self, obs, action, reward, obs_new):
        self.buffer.add_xp(obs, action, reward, obs_new)

    def critic(self, *, observations, actions, use_target_network=False):
        x = tf.concat( [observations, actions] , axis=1)
        network = self.critic_t if use_target_network else self.critic_net
        return network(x)

    def train_critic(self, sars):
        obs1, actions, rewards, obs2 = sars

        expected_actions = self.compute_actions(obs2, use_target_network=True)
        expected_value = self.critic(observations=obs2,
                                     actions=expected_actions,
                                     use_target_network=True)

        x_train = np.concatenate([obs1, actions], axis=1)
        y_train = rewards + self.future_discount * expected_value

        self.critic_net.fit(x_train, y_train, verbose=0)

    def train_actor(self, sars):

        obs1, actions, rewards, obs2 = sars
        with tf.GradientTape() as tape:
            would_do_actions = self.compute_actions(obs1)
            score = tf.reduce_mean( self.critic( observations=obs1, actions=would_do_actions ) )
            inverted = - score

        # tf optimizer follows negative of the provided gradients.
        # For this reason we provide negative gradient of the score -
        # it will result in positive gradient being followed.
        grads = tape.gradient( inverted, self.actor.trainable_weights )
        self.optimizer.apply_gradients( zip(grads, self.actor.trainable_weights) )




