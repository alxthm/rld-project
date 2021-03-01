import os
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from kaggle_environments.envs.rps.utils import get_score
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo_model import PPO

T_horizon = 20


class RockPaperScissorsEnv:
    def __init__(self, opponent: Callable):
        self.configuration = SimpleNamespace(**{'episodeSteps': 10, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3,
                                                'tieRewardThreshold': 20, 'agentTimeout': 60})
        self.opponent = opponent
        self.obs_right = None
        self.reward_left = 0
        self.reward_right = 0
        self.t = 0

    def reset(self):
        obs_left = SimpleNamespace(**{'remainingOverageTime': 60, 'step': 0, 'reward': 0})
        self.obs_right = SimpleNamespace(**{'remainingOverageTime': 60, 'step': 0, 'reward': 0})
        self.reward_left = 0
        self.reward_right = 0
        self.t = 0
        return np.zeros(7)

    def step(self, a: int):
        action_left = a
        action_right = self.opponent(self.obs_right, self.configuration)
        score = get_score(action_left, action_right)
        self.reward_left += score
        self.reward_right -= score
        self.t += 1
        obs_left = SimpleNamespace(**{'remainingOverageTime': 60, 'step': self.t, 'reward': self.reward_left,
                                      'lastOpponentAction': action_right})
        self.obs_right = SimpleNamespace(**{'remainingOverageTime': 60, 'step': self.t, 'reward': self.reward_right,
                                            'lastOpponentAction': action_left})
        done = (self.t == 1000)
        obs = np.zeros(7)
        obs[action_left] = 1
        obs[action_right] = 1
        obs[-1] = self.t
        return obs, score, done


def train_one_ep_against(opponent_name: str, env, model, writer, current_global_step):
    score = 0.
    h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
    s = env.reset()
    done = False
    global_step = current_global_step

    while not done:
        # gather a batch of T_horizon transitions in order to do a PPO update step
        for t in range(T_horizon):
            h_in = h_out
            logits, h_out = model.pi(torch.from_numpy(s).float(), h_in)
            logits = logits.view(-1)
            m = Categorical(logits=logits)
            a = m.sample().item()
            s_prime, r, done = env.step(a)

            model.put_data((s, a, r, s_prime, logits.detach().numpy(), h_in, h_out, done))
            s = s_prime

            score += r
            if done:
                break

        logs = model.train_net()
        global_step += 1
        if global_step % 20 == 0:
            for tag, value in logs.items():
                writer.add_scalar(f'{tag}_{opponent_name}', value, global_step)

    return score, global_step


def train(n_runs):
    opponents = {
        # 'white_belt/all_paper': all_paper.constant_play_agent_1,
        # 'white_belt/all_rock': all_rock.constant_play_agent_0,
        # 'white_belt/all_scissors': all_scissors.constant_play_agent_2,
        # 'white_belt/mirror': mirror.mirror_opponent_agent,
        # 'white_belt/reactionary': reactionary.reactionary,
        # 'white_belt/de_bruijn': de_bruijn.kaggle_agent,
        # 'blue_belt/transition_matrix': transition_matrix.transition_agent,
        # 'blue_belt/not_so_markov': not_so_markov.markov_agent,
        # 'blue_belt/decision_tree': decision_tree.agent,
        # 'black_belt/multi_armed_bandit_v15': multi_armed_bandit_v15.multi_armed_bandit_agent,
        'black_belt/testing_please_ignore': testing_please_ignore.run,
    }

    # create tensorboard writer and save current file
    log_dir = project_dir / f'runs/levi/{datetime.now().strftime(f"%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    shutil.copy2(project_dir / 'dojo/my_little_dojo/levi.py', log_dir)
    shutil.copy2(__file__, log_dir)

    for opponent_name, opponent in opponents.items():
        print(f'Training against {opponent_name}')
        model = PPO()
        env = RockPaperScissorsEnv(opponent)
        global_step = 0
        for n_epi in tqdm(range(n_runs)):
            score, global_step = train_one_ep_against(opponent_name, env, model, writer, global_step)

            # log to tensorboard
            writer.add_scalar(f'score_{opponent_name}', score, n_epi)
            if score > 995:
                break  # consider the task solved

            if n_epi % 300 == 0 and n_epi > 0:
                # save model weights
                os.makedirs(os.path.dirname(log_dir / f'model_{opponent_name}_{n_epi}.pth'), exist_ok=True)
                torch.save(model.state_dict(), log_dir / f'model_{opponent_name}_{n_epi}.pth')

        # save model weights
        os.makedirs(os.path.dirname(log_dir / f'model_{opponent_name}.pth'), exist_ok=True)
        torch.save(model.state_dict(), log_dir / f'model_{opponent_name}.pth')


def train_against_all(n_runs):
    opponents = {
        'white_belt/all_paper': all_paper.constant_play_agent_1,
        'white_belt/all_rock': all_rock.constant_play_agent_0,
        'white_belt/all_scissors': all_scissors.constant_play_agent_2,
        'white_belt/mirror': mirror.mirror_opponent_agent,
        'white_belt/reactionary': reactionary.reactionary,
        'white_belt/de_bruijn': de_bruijn.kaggle_agent,
    }

    # create tensorboard writer and save current file
    log_dir = project_dir / f'runs/levi/all_{datetime.now().strftime(f"%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    shutil.copy2(project_dir / 'dojo/my_little_dojo/levi.py', log_dir)
    shutil.copy2(__file__, log_dir)

    print(f'Training against agents {opponents.keys()}')
    model = PPO()
    global_step = 0

    for n_epi in tqdm(range(n_runs)):
        # pick a random opponent
        opponent_id = np.random.randint(len(opponents))
        opponent_name = list(opponents.keys())[opponent_id]
        opponent = opponents[opponent_name]

        # log which opponent was chosen
        for name in opponents.keys():
            if name == opponent_name:
                writer.add_scalar(f'chose_{opponent_name}', 1, n_epi)
            else:
                writer.add_scalar(f'chose_{opponent_name}', 0, n_epi)

        env = RockPaperScissorsEnv(opponent)
        score, global_step = train_one_ep_against('all', env, model, writer, global_step)

        # log to tensorboard
        writer.add_scalar(f'score_all', score, n_epi)
        writer.add_scalar(f'score_all_{opponent_name}', score, n_epi)

        if n_epi % 300 == 0 and n_epi > 0:
            # save model weights
            os.makedirs(os.path.dirname(log_dir / f'model_{n_epi}.pth'), exist_ok=True)
            torch.save(model.state_dict(), log_dir / f'model_{n_epi}.pth')

    # save model weights
    os.makedirs(os.path.dirname(log_dir / f'model_all.pth'), exist_ok=True)
    torch.save(model.state_dict(), log_dir / f'model_all.pth')


if __name__ == '__main__':
    project_dir = Path(os.path.realpath(__file__)).parent
    sys.path.append(str(project_dir))
    from dojo.white_belt import all_paper, all_scissors, all_rock, reactionary, mirror, de_bruijn
    from dojo.blue_belt import decision_tree, transition_matrix, not_so_markov
    from dojo.black_belt import testing_please_ignore, multi_armed_bandit_v15

    train(n_runs=10000)
    # train_against_all(n_runs=30000)
