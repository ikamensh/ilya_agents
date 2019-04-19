from ilya_agents.ddpg import Ddpg

# from pprint import pprint
#
# for e in list(gym.envs.registry.all() ):
#     if 'Conti' in e._env_name:
#         print(e)

from ilya_agents.ez_envs.int_env import Integrating

env = Integrating()
# env = gym.make('MountainCarContinuous-v0')


print(env.action_space, env.observation_space)

agent = Ddpg(obs_space=env.observation_space, action_space=env.action_space, buffer_size=int(1e4))
agent.tracking_speed = 0.99

def train_episode(episode):
    total_reward = 0
    n_steps = 0
    obs, done = env.reset(), False

    while not done:
        action = agent.env_action(obs)
        obs_new, reward, done, info = env.step(action)
        n_steps += 1
        total_reward += reward

        agent.add_xp(obs, action, reward, obs_new)
        agent.learn(batch_size=32)
        obs = obs_new

    print(episode, f"Train: steps: {n_steps} reward: {total_reward:.2g}")
    return total_reward


def trial_episode(episode):
    total_reward = 0
    n_steps = 0
    obs, done = env.reset(), False
    with agent.no_exploration():
        while not done:
            env.render()
            action = agent.env_action(obs)
            obs_new, reward, done, info = env.step(action)
            n_steps += 1
            total_reward += reward

            obs = obs_new

    print(episode, f"Test: steps: {n_steps} reward: {total_reward:.2f}")
    return total_reward


for episode in range(100):

    train_episode(episode)
    if episode % 5 == 0:
        trial_episode(episode)


