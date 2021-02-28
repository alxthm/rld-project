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
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parents[0]


def sanitize_action(action):
    return -1 if action is None else int(action)


def evaluate(agents, num_episodes, configuration={}, steps=[]):
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
    outcomes, r, a_left, a_right = evaluate([agent, baseline], num_episodes)
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
        baseline_name = baseline.split('/')[-1].split('.py')[0]
        # baseline_name = re.search(r"(?<=\\).+(?=\.)", baseline).group(0)
        df_stats.append(pd.DataFrame({
            'ep': i, 't': [t * 10 for t in range(len(r[i]))], 'rewards': r[i], 'actions_left': a_left[i],
            'actions_right': a_right[i], 'opponent': baseline_name
        }))
    elapsed = datetime.now() - start
    return baseline, won, lost, tie, elapsed, cum_score, df_stats


def eval_agent_against_baselines(agent: str, baselines: List[str], num_episodes: int):
    """

    Args:
        agent: path of the agent python file
        baselines: list of paths
        num_episodes: number of episodes to run (each episodes consisting of 1000 matches, with a winner and a loser
            at the end)

    Returns:

    """
    # baselines_names = [re.search(r"(?<=\\).+(?=\.)", baseline).group(0) for baseline in baselines] # on Windows
    baselines_names = [baseline.split('/')[-1].split('.py')[0] for baseline in baselines]  # on Unix systems
    df = pd.DataFrame(
        columns=['wins', 'ties', 'loses', 'total time', 'avg. score'],
        index=baselines_names
    )

    pool = pymp.Pool()
    match_settings = [[agent, baseline, num_episodes] for baseline in baselines]

    results = []
    # for m in tqdm(range(len(match_settings))):
    #     content = get_result(match_settings[m])
    for content in tqdm(pool.imap_unordered(get_result, match_settings), total=len(match_settings)):
        results.append(content)
    pool.close()

    df_all = []
    for baseline_agent, won, lost, tie, elapsed, avg_score, df_stats in results:
        # baseline_name = re.search(r"(?<=\\).+(?=\.)", baseline_agent).group(0)
        baseline_name = baseline_agent.split('/')[-1].split('.py')[0]
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


def plot_figures(df_results, df_all, log_dir, agent_name=None):
    df_results = df_results.reset_index().rename(columns={'index': 'opponent'}).melt(id_vars=['opponent', 'total time'])
    df_results.value = df_results.value.astype(int)
    fig1 = sns.catplot(
        data=df_results[df_results.variable != 'avg. score'], kind="bar",
        x="variable", y="value", hue="opponent",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    fig2 = sns.relplot(x='t', y='rewards', col='opponent', hue='opponent', col_wrap=4, kind='line', data=df_all)

    if agent_name == 'giorno':
        tmp_log_path = log_dir.parent.parent / 'tmp'
        freq_df = pd.read_csv(tmp_log_path / 'giorno_freq_df.csv', index_col=0)
        freq_df.to_csv(log_dir / 'giorno_freq_df.csv')

        freq_df = freq_df.reset_index().rename(columns={'index': 'timestep'})
        freq_df = freq_df.melt(id_vars=["timestep"])
        fig3 = sns.relplot(x='timestep', y='value', col='variable', hue='variable', col_wrap=4, kind='line',
                           data=freq_df)
        fig3.savefig(log_dir / 'agents_chosen_frequency.png')

    fig1.savefig(log_dir / 'results.png')
    fig2.savefig(log_dir / 'full_history.png')


def plot_levi_figures(df_results, log_dir):
    df_results = df_results.reset_index().rename(columns={'index': 'opponent', 'avg. score': 'avg_score'})
    df_results['wins_minus_loses'] = (df_results.wins - df_results.loses).astype(int)
    df_results.avg_score = df_results.avg_score.astype(float)

    plt.figure()
    ax1 = sns.heatmap(df_results.pivot('trained_against', 'opponent', 'avg_score'))
    ax1.set_title('Average score')
    plt.tight_layout()
    ax1.figure.savefig(log_dir / 'heatmap_avg_score.png')

    plt.figure()
    ax2 = sns.heatmap(df_results.pivot('trained_against', 'opponent', 'wins_minus_loses'))
    ax2.set_title('Wins minus loses')
    plt.tight_layout()
    ax2.figure.savefig(log_dir / 'heatmap_wins_minus_loses.png')


def evaluate_multi_levi():
    opponent_dojo = 'white'
    time_stamp = datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = project_dir / f'runs/csv_results/multi-levi-{opponent_dojo}-{time_stamp}'
    save_conf(log_dir, 'levi')

    opponents = [os.path.join(f'dojo/{opponent_dojo}_belt', agent) for agent in
                 os.listdir(f'dojo/{opponent_dojo}_belt')]
    opponents = [o for o in opponents if '__pycache__' not in o]

    # names of the agent it was trained on
    levi_agents = [agent.split('.')[0] for agent in os.listdir(f'dojo/my_little_dojo/levi')]

    df_list = []
    for levi_agent in levi_agents:
        my_agent = f'dojo/my_little_dojo/levi/{levi_agent}.py'

        print(f'Evaluating levi/{levi_agent}')
        df, _ = eval_agent_against_baselines(my_agent, opponents, num_episodes=10)
        df['trained_against'] = levi_agent
        df_list.append(df)
        print(df)

    # save eval results
    df = pd.concat(df_list)
    df.reset_index().to_csv(log_dir / 'results.csv', index=False)
    plot_levi_figures(df, log_dir)


def main():
    agent_name = 'levi/all'
    opponent_dojo = 'white'

    # save conf and code
    time_stamp = datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = project_dir / f'runs/csv_results/{agent_name}-{opponent_dojo}-{time_stamp}'
    os.makedirs(log_dir, exist_ok=True)
    if not ('levi' in agent_name):  # no conf file for levi
        save_conf(log_dir, agent_name)

    my_agent = f'dojo/my_little_dojo/{agent_name}.py'
    opponents = [os.path.join(f'dojo/{opponent_dojo}_belt', agent) for agent in
                 os.listdir(f'dojo/{opponent_dojo}_belt')]
    opponents = [o for o in opponents if '__pycache__' not in o]

    df, df_all = eval_agent_against_baselines(my_agent, opponents, num_episodes=10)
    print(df)

    # save eval results
    df.reset_index().to_csv(log_dir / 'results.csv', index=False)
    df_all.to_csv(log_dir / 'full_history.csv', index=False)
    plot_figures(df, df_all, log_dir, agent_name)


if __name__ == '__main__':
    main()
    # evaluate_multi_levi()
