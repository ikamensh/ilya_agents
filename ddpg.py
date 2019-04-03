import random
from collections import deque
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, Sequential
from gym import spaces
from replay_buffer import ReplayBuffer

buffer = deque


class Ddpg:

    def __init__(self, action_space: spaces.Box, obs_space: spaces.Space,
                 buffer_size, epsilon = 0.1, critic_lr = None, actor_lr = None):

        assert len(action_space.shape) == 1
        assert len(obs_space.shape) == 1
        self.epsilon = epsilon

        self.scale = action_space.high - action_space.low
        self.offset = action_space.low
        self.action_space = action_space

        n_actions = action_space.shape[0]
        n_inputs = obs_space.shape[0]
        critic = Sequential()
        critic.add(layers.Dense(32, input_shape=[n_inputs + n_actions], activation='relu'))
        critic.add(layers.Dense(1))
        critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr or 1e-3)
        critic.compile(optimizer=critic_optimizer, loss='mse')
        self.critic_net = critic
        # self.critic_t = copy.deepcopy(critic)

        actor = Sequential()
        actor.add(layers.Dense(32, input_shape=[n_inputs]))
        actor.add(layers.Dense(n_actions, activation='sigmoid'))
        self.actor = actor
        # self.actor_t = copy.deepcopy(actor)

        self.optimizer = tf.optimizers.Adam()

        self.buffer = ReplayBuffer(maxlen=buffer_size)


    def act(self, observation, explore = False):
        assert hasattr(observation, 'shape'), f"{observation} must be a np / tf tensor"
        assert len(observation.shape) in [1,2], f"must be at most 2d, found: {observation.shape}"

        if explore and self.epsilon > random.random():
            return self.action_space.sample()

        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0)

        return self.scale * self.actor(observation) + self.offset


    def learn(self, *, batch_size):
        sars = self.buffer.get_batch(batch_size)
        self.train_critic(sars)
        self.train_actor(sars)

    def add_xp(self, obs, action, reward, obs_new):
        self.buffer.add_xp(obs, action, reward, obs_new)

    def critic(self, *, observations, actions):
        x = tf.concat( [observations, actions] , axis=1)
        return self.critic_net(x)

    def train_critic(self, sars):
        obs1, actions, rewards, obs2 = sars

        expected_actions = self.act(obs2)
        expected_value = self.critic(observations=obs2, actions=expected_actions)
        discount = 0.99

        x_train = np.concatenate([obs1, actions], axis=1)
        y_train = rewards + discount * expected_value

        self.critic_net.fit(x_train, y_train, verbose=0)

    def train_actor(self, sars):

        obs1, actions, rewards, obs2 = sars
        with tf.GradientTape() as tape:
            would_do_actions = self.actor(obs1)
            score = tf.reduce_mean( self.critic( observations=obs1, actions=would_do_actions ) )

        grads = tape.gradient( score, self.actor.trainable_weights )
        self.optimizer.apply_gradients( zip(grads, self.actor.trainable_weights) )














# dummy_input = np.ones([1,1])
# print(critic(dummy_input), actor(dummy_input))
