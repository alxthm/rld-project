from types import SimpleNamespace
from typing import Callable

from kaggle_environments import make
from kaggle_environments.envs.rps.utils import get_score

from dojo.white_belt import statistical, counter_reactionary, all_paper
from dojo.my_little_dojo import saitama, jotaro, giorno


def test_env(agent_left: Callable, agent_right: Callable, steps: int, verbose: bool):
    configuration = SimpleNamespace(**{'episodeSteps': 10, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3,
                                       'tieRewardThreshold': 20, 'agentTimeout': 60})

    obs_left = SimpleNamespace(**{'remainingOverageTime': 60, 'step': 0, 'reward': 0})
    obs_right = SimpleNamespace(**{'remainingOverageTime': 60, 'step': 0, 'reward': 0})
    reward_left = 0
    reward_right = 0
    for step in range(steps):
        if step > 0:
            obs_left = SimpleNamespace(**{'remainingOverageTime': 60, 'step': step, 'reward': reward_left,
                                          'lastOpponentAction': action_right})
            obs_right = SimpleNamespace(**{'remainingOverageTime': 60, 'step': step, 'reward': reward_right,
                                           'lastOpponentAction': action_left})
        action_left = agent_left(obs_left, configuration)
        action_right = agent_right(obs_right, configuration)
        score = get_score(action_left, action_right)
        reward_left += score
        reward_right -= score
        if verbose:
            print(f'step {step} - {action_left} vs {action_right} : score {score}')
    print(f'Final score: {reward_left} - {reward_right}')


def evaluate_with_debug(agent, configuration={}, steps=[], debug=False, num_episodes=1):
    e = make('rps', configuration, steps, debug=debug)
    rewards = [[]] * num_episodes
    for i in range(num_episodes):
        last_state = e.run([agent, 'statistical'])[-1]
        rewards[i] = [state.reward for state in last_state]
    return rewards


def single_run(agent, config):
    env = make('rps', configuration=config, debug=True)
    env.run([agent, 'statistical'])
    print(env.render(mode='ansi'))


if __name__ == '__main__':
    # --- Single or multiple runs with kaggle env
    config = {'episodeSteps': 1000}
    my_agent = 'dojo/my_little_dojo/saitama_tabular.py'
    # single_run(my_agent, config)
    # evaluate_with_debug(my_agent, config, debug=True, num_episodes=2)

    # --- Test and debug with test environment
    test_env(giorno.banditception, counter_reactionary.counter_reactionary, steps=1000, verbose=True)
