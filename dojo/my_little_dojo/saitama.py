import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.rps.utils import get_score
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TabularAgent:
    def __init__(self, alpha, gamma, exploration_strategy, algo, dim_a, eps_0=None, eps_decay=None, tau=None):
        self.Q = defaultdict(lambda: np.zeros(dim_a))
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.algorithm = algo
        self.eps_0 = eps_0
        self.eps_decay = eps_decay
        self.episode = 0
        self.tau = tau
        self.dim_a = dim_a
        self.training = True

    def act(self, obs):
        obs = str(obs)
        if self.exploration_strategy == 'epsilon-greedy':
            eps = self.eps_0 * self.eps_decay
            if np.random.random() < eps:
                return np.random.randint(0, self.dim_a)
            else:
                return np.argmax(self.Q[obs])

        elif self.exploration_strategy == 'boltzmann':
            probas = [np.exp(self.Q[obs][a] / self.tau) / np.sum(np.exp(self.Q[obs] / self.tau)) for a in
                      range(self.dim_a)]
            return np.random.choice(self.dim_a, p=probas)

    def learn(self, s, a, r, s_prime, a_prime=None):
        s = str(s)
        s_prime = str(s_prime)
        if self.algorithm == 'Q-learning':
            self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s][a])
        elif self.algorithm == 'SARSA':
            self.Q[s][a] += self.alpha * (r + self.gamma * self.Q[s_prime][a_prime] - self.Q[s][a])


# def train_agent():
#     project_dir = Path(__file__).resolve().parents[2]
#     conf = OmegaConf.load(project_dir / 'dojo/my_little_dojo/saitama.yaml')
#     log_dir = project_dir / f'runs/tabular/{datetime.now().strftime(f"%Y%m%d-%H%M%S")}'
#     writer = SummaryWriter(log_dir)
#
#     # --- Hyper params
#
#     train_episodes = 100
#     opponents = [project_dir / f'dojo/white_belt/{opponent}' for opponent in
#                  os.listdir(project_dir / 'dojo/white_belt')]
#
#     agent = TabularAgent(alpha=conf.alpha, gamma=conf.discount, exploration_strategy=conf.exploration,
#                          algo=conf.agent_strategy, dim_a=3, eps_0=conf.epsilon)
#
#     env = make('rps')
#     global_step = 0
#     for ep in tqdm(range(train_episodes)):
#         # do an episode against a random opponent
#         opponent = opponents[np.random.choice(len(opponents))]
#         trainer = env.train([None, opponent])
#         print(f'ep {ep} - opponent {opponent}')
#
#         observation = trainer.reset()
#         action_buffer = []
#         for t in range(1000):
#             # save last opponent action
#             if observation.step > 0:
#                 action_buffer.append(observation.lastOpponentAction)
#
#             if len(action_buffer) < history_length:
#                 # play random actions at the beginning
#                 action = np.random.randint(3)
#             else:
#                 # learning step
#                 s = action_buffer[-history_length - 1:-1]
#                 a = action_buffer[-2]
#                 r = get_score(a, observation.lastOpponentAction)
#                 s_prime = action_buffer[-history_length:]
#                 agent.learn(s, a, r, s_prime)
#
#                 action = int(agent.act(s_prime))
#
#             action_buffer.append(action)
#             observation, reward, _, _ = trainer.step(action)
#
#             if global_step % 10 == 0:
#                 writer.add_scalar('reward', reward, global_step)
#             global_step += 1


conf = OmegaConf.load('dojo/my_little_dojo/saitama.yaml')
k = conf.history_length
action_buffer = []
agent = TabularAgent(alpha=conf.alpha, gamma=conf.discount, exploration_strategy=conf.exploration,
                     algo=conf.agent_strategy, dim_a=3, eps_0=conf.epsilon, eps_decay=conf.epsilon_decay)


def act_func(observation, configuration=None):
    global agent
    global action_buffer
    global k  # number of past actions to consider

    # save last opponent action
    if observation.step > 0:
        action_buffer.append(observation.lastOpponentAction)

    if len(action_buffer) < k + 2:
        # play random actions at the beginning
        action = np.random.randint(3)
    else:
        # observe last actions
        s = action_buffer[-k - 2:-2]
        a = action_buffer[-2]
        r = get_score(a, observation.lastOpponentAction)
        s_prime = action_buffer[-k:]
        agent.learn(s, a, r, s_prime)
        action = int(agent.act(action_buffer[-k:]))

    action_buffer.append(action)
    return action
