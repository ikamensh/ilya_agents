from ilya_ezplot import Metric

def train_episode(agent, env):
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

    return total_reward


def trial_episode(agent, env):
    total_reward = 0
    n_steps = 0
    obs, done = env.reset(), False
    with agent.no_exploration():
        while not done:
            # env.render()
            action = agent.env_action(obs)
            obs_new, reward, done, info = env.step(action)
            n_steps += 1
            total_reward += reward

            obs = obs_new

    return total_reward


def one_try(env, agent_constructor, episodes):
    agent = agent_constructor(env)
    training_reward = Metric('episode', 'training reward')
    test_reward = Metric('episode', 'test reward')
    for episode in range(episodes):

        r_train = train_episode(agent, env)
        print(episode, f"Train: reward: {r_train:.2g}")

        training_reward.add_record(episode, r_train)
        if episode % 3 == 0:
            r_test = trial_episode(agent, env)
            print(episode, f"Test: reward: {r_test:.2f}")
            test_reward.add_record(episode, r_test)

    return training_reward, test_reward


def test_env(env, agent_constructor, n_tries, n_episodes):
    training = []
    test = []
    for i in range(n_tries):
        train_r_metric, test_r_metric = one_try(env, agent_constructor, n_episodes)
        training.append(train_r_metric)
        test.append(test_r_metric)
    return sum(training), sum(test)