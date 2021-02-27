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

from dojo.my_little_dojo.levi import PPO

T_horizon = 20
n_runs = 10000  # how many matches to do against each agent for training


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


def train():
    project_dir = Path(os.path.realpath(__file__)).parent
    sys.path.append(str(project_dir))
    from dojo.white_belt import all_paper, all_scissors, all_rock, reactionary, mirror, statistical
    from dojo.blue_belt import decision_tree, transition_matrix, not_so_markov
    from dojo.black_belt import testing_please_ignore, multi_armed_bandit_v15

    opponents = {
        # 'white_belt/all_paper': all_paper.constant_play_agent_1,
        # 'white_belt/all_rock': all_rock.constant_play_agent_0,
        # 'white_belt/all_scissors': all_scissors.constant_play_agent_2,
        # 'white_belt/mirror': mirror.mirror_opponent_agent,
        'white_belt/reactionary': reactionary.reactionary,
        # 'blue_belt/transition_matrix': transition_matrix.transition_agent,
        # 'blue_belt/not_so_markov': not_so_markov.markov_agent,
        # 'blue_belt/decision_tree': decision_tree.agent,
        # 'black_belt/multi_armed_bandit_v15': multi_armed_bandit_v15.multi_armed_bandit_agent,
        # 'black_belt/testing_please_ignore': testing_please_ignore.run,
    }

    # create tensorboard writer and save current file
    log_dir = project_dir / f'runs/levi/{datetime.now().strftime(f"%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    shutil.copy2(__file__, log_dir)

    # create
    for opponent_name, opponent in opponents.items():
        print(f'Training against {opponent_name}')
        model = PPO()
        env = RockPaperScissorsEnv(opponent)
        for n_epi in range(n_runs):
            score = 0.
            h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
            s = env.reset()
            done = False

            while not done:
                # gather a batch of T_horizon transitions in order to do a PPO update step
                for t in range(T_horizon):
                    h_in = h_out
                    prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done = env.step(a)

                    model.put_data((s, a, r, s_prime, prob[a].item(), h_in, h_out, done))
                    s = s_prime

                    score += r
                    if done:
                        break

                model.train_net()

            # log to tensorboard
            writer.add_scalar(f'score_{opponent_name}', score, n_epi)
            if score > 995:
                break  # consider the task solved

            if n_epi % 100 == 0 and n_epi > 0:
                # save model weights
                os.makedirs(os.path.dirname(log_dir / f'model_{opponent_name}_{n_epi}.pth'), exist_ok=True)
                torch.save(model.state_dict(), log_dir / f'model_{opponent_name}_{n_epi}.pth')

        # save model weights
        os.makedirs(os.path.dirname(log_dir / f'model_{opponent_name}.pth'), exist_ok=True)
        torch.save(model.state_dict(), log_dir / f'model_{opponent_name}.pth')


if __name__ == '__main__':
    train()
