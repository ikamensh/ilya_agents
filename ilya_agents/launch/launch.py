import time
import gym
from ilya_ezplot import plot_group

from ilya_agents.agents.ddpg import Ddpg
from ilya_agents.ez_envs.int_env import Integrating
from ilya_agents.ez_envs.ez_env import Ez
from ilya_agents.ez_envs.exp_range_env import Exponential
from ilya_agents.launch.lib import test_env


envs = [Integrating(), Ez(), Exponential(), gym.make('MountainCarContinuous-v0')]


def agent_constructor(env):
    return Ddpg(obs_space=env.observation_space,
                 action_space=env.action_space,
                 buffer_size=int(1e5))


def maping(env):
    try:
        return test_env(env, agent_constructor, n_tries=2, n_episodes=4)
    except Exception as e:
        print(e, 'In mapping')
        return None
    finally:
        time.sleep(1)


if __name__ == "__main__":
    from concurrent import futures

    executor = futures.ProcessPoolExecutor()
    todos = {}
    for env in envs:
        todos[executor.submit(maping, env)] = env

    scores = {}
    for future in futures.as_completed(todos):
        scores[ str(todos[future]) ] = future.result()


    for k, v in scores.items():
        plot_group({'train': k, 'test': v}, 'plots', name=f"{Ddpg.__name__} vs {k}")







