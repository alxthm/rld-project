import multiprocessing as pymp
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import re
import kaggle_environments
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[0]


def sanitize_action(action):
    return -1 if action is None else int(action)


def evaluate(agents, configuration={}, steps=[], num_episodes=1):
    env = kaggle_environments.make('rps', configuration, steps, debug=False)
    final_rewards = []
    rewards = []
    actions_left = []
    actions_right = []
    for i in range(num_episodes):
        states = env.run(agents)
        final_rewards.append([state.reward for state in states[-1]])  # 2 final rewards of the episode
        # all rewards and actions of the left agent
        rewards.append([state_t[0].reward for t, state_t in enumerate(states) if t % 10 == 0])
        actions_left.append([state_t[0].action for t, state_t in enumerate(states) if t % 10 == 0])
        actions_right.append([sanitize_action(state_t[1].action) for t, state_t in enumerate(states) if t % 10 == 0])
    return final_rewards, rewards, actions_left, actions_right


def get_result(match_setting: Tuple[str, str, int]):
    start = datetime.now()
    agent, baseline, num_episodes = match_setting
    outcomes, r, a_left, a_right = evaluate([agent, baseline], num_episodes=num_episodes)
    won, lost, tie, cum_score = 0, 0, 0, 0.
    df_stats = []
    for i, outcome in enumerate(outcomes):
        score = outcome[0]
        if score > 0:
            won += 1
        elif score < 0:
            lost += 1
        else:
            tie += 1
        cum_score += score
        # baseline_name = baseline.split('/')[-1].split('.py')[0]
        baseline_name = re.search(r"(?<=\\).+(?=\.)", baseline).group(0)
        df_stats.append(pd.DataFrame({
            'ep': i, 't': [t * 10 for t in range(len(r[i]))], 'rewards': r[i], 'actions_left': a_left[i],
            'actions_right': a_right[i], 'opponent': baseline_name
        }))
    elapsed = datetime.now() - start
    return baseline, won, lost, tie, elapsed, cum_score, df_stats


def eval_agent_against_baselines(agent: str, baselines: List[str], num_episodes=10):
    """

    Args:
        agent: path of the agent python file
        baselines: list of paths
        num_episodes: number of episodes to run (each episodes consisting of 1000 matches, with a winner and a loser
            at the end)

    Returns:

    """
    baselines_names = [re.search(r"(?<=\\).+(?=\.)", baseline).group(0) for baseline in baselines]
    # baselines_names = [baseline.split('/')[-1].split('.py')[0] for baseline in baselines]
    df = pd.DataFrame(
        columns=['wins', 'ties', 'loses', 'total time', 'avg. score'],
        index=baselines_names
    )

    pool = pymp.Pool()
    match_settings = [[agent, baseline, num_episodes] for baseline in baselines]

    results = []
    for content in tqdm(pool.imap_unordered(get_result, match_settings), total=len(match_settings)):
        results.append(content)
    pool.close()

    df_all = []
    for baseline_agent, won, lost, tie, elapsed, avg_score, df_stats in results:
        baseline_name = re.search(r"(?<=\\).+(?=\.)", baseline_agent).group(0)
        df.loc[baseline_name, 'wins'] = won
        df.loc[baseline_name, 'ties'] = tie
        df.loc[baseline_name, 'loses'] = lost
        df.loc[baseline_name, 'total time'] = elapsed
        df.loc[baseline_name, 'avg. score'] = avg_score
        df_all += df_stats

    return df, pd.concat(df_all)


def save_conf(log_dir, agent_name: str):
    conf = OmegaConf.load(project_dir / f'dojo/my_little_dojo/{agent_name}.yaml')
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/conf.yaml', 'w') as f:
        OmegaConf.save(conf, f)


def plot_figures(df_results, df_all, log_dir):
    df_results = df_results.reset_index().rename(columns={'index': 'opponent'}).melt(id_vars=['opponent', 'total time'])
    df_results.value = df_results.value.astype(int)
    fig1 = sns.catplot(
        data=df_results[df_results.variable != 'avg. score'], kind="bar",
        x="variable", y="value", hue="opponent",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    fig2 = sns.relplot(x='t', y='rewards', col='opponent', hue='opponent', col_wrap=4, kind='line', data=df_all)
    fig1.savefig(log_dir / 'results.png')
    fig2.savefig(log_dir / 'full_history.png')


def main():
    agent_name = 'giorno'
    opponent_dojo = 'blue'

    # save conf and code
    time_stamp = datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = project_dir / f'runs/csv_results/{agent_name}-{opponent_dojo}-{time_stamp}'
    save_conf(log_dir, agent_name)

    # my_agent = 'dojo/black_belt/greenberg.py'
    my_agent = f'dojo/my_little_dojo/{agent_name}.py'
    opponents = [os.path.join(f'dojo/{opponent_dojo}_belt', agent) for agent in
                 os.listdir(f'dojo/{opponent_dojo}_belt')]

    # remnant file in opponents to pop
    try:
        opponents.remove('dojo/white_belt\\__pycache__')
    except ValueError:
        print("no '__pycache__' found in opponents")

    df, df_all = eval_agent_against_baselines(my_agent, opponents, num_episodes=10)
    print(df)

    # save eval results
    df.reset_index().to_csv(log_dir / 'results.csv', index=False)
    df_all.to_csv(log_dir / 'full_history.csv', index=False)
    plot_figures(df, df_all, log_dir)


if __name__ == '__main__':
    main()
