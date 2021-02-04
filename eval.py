import multiprocessing as pymp
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import kaggle_environments
import pandas as pd
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[0]


def get_result(match_setting: Tuple[str, str, int]):
    start = datetime.now()
    agent, baseline, num_episodes = match_setting
    outcomes = kaggle_environments.evaluate('rps', [agent, baseline], num_episodes=num_episodes)
    won, lost, tie, cum_score = 0, 0, 0, 0.
    for outcome in outcomes:
        score = outcome[0]
        if score > 0:
            won += 1
        elif score < 0:
            lost += 1
        else:
            tie += 1
        cum_score += score
    elapsed = datetime.now() - start
    return baseline, won, lost, tie, elapsed, cum_score


def eval_agent_against_baselines(agent: str, baselines: List[str], num_episodes=10) -> pd.DataFrame:
    """

    Args:
        agent: path of the agent python file
        baselines: list of paths
        num_episodes: number of episodes to run (each episodes consisting of 1000 matches, with a winner and a loser
            at the end)

    Returns:

    """
    df = pd.DataFrame(
        columns=['wins', 'loses', 'ties', 'total time', 'avg. score'],
        index=baselines
    )

    pool = pymp.Pool()
    match_settings = [[agent, baseline, num_episodes] for baseline in baselines]

    results = []
    for content in tqdm(pool.imap_unordered(get_result, match_settings), total=len(match_settings)):
        results.append(content)
    pool.close()

    for baseline_agent, won, lost, tie, elapsed, avg_score in results:
        df.loc[baseline_agent, 'wins'] = won
        df.loc[baseline_agent, 'loses'] = lost
        df.loc[baseline_agent, 'ties'] = tie
        df.loc[baseline_agent, 'total time'] = elapsed
        df.loc[baseline_agent, 'avg. score'] = avg_score

    agent_name = agent.split('/')[-1].split('.')[0]
    df.to_csv(project_dir / f'runs/csv_results/{agent_name}-{datetime.now().strftime(f"%Y%m%d-%H%M%S")}.csv')
    return df


def main():
    # my_agent = 'dojo/black_belt/greenberg.py'
    my_agent = 'dojo/my_little_dojo/saitama_tabular.py'
    opponent_dojo = 'white'
    opponents = [os.path.join(f'dojo/{opponent_dojo}_belt', agent) for agent in
                 os.listdir(f'dojo/{opponent_dojo}_belt')]
    print(eval_agent_against_baselines(my_agent, opponents))


if __name__ == '__main__':
    main()
