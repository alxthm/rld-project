from kaggle_environments import make


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
    config = {'episodeSteps': 10}
    my_agent = 'dojo/my_little_dojo/saitama_tabular.py'
    single_run(my_agent, config)
    # evaluate_with_debug(my_agent, config, debug=True, num_episodes=2)
