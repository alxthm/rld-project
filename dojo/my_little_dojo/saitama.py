import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.rps.utils import get_score
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


def train_agent():
    project_dir = Path(__file__).resolve().parents[2]
    log_dir = project_dir / f'runs/tabular/{datetime.now().strftime(f"%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir)

    # --- Hyper params
    discount = 0.99
    history_length = 5
    agent_strategy = 'Q-learning'
    learning_rate = 0.1
    exploration = 'epsilon-greedy'
    epsilon = 0.5
    # exploration = 'boltzmann'
    tau = 0.01

    train_episodes = 100
    opponents = [project_dir / f'dojo/white_belt/{opponent}' for opponent in
                 os.listdir(project_dir / 'dojo/white_belt')]

    agent = TabularAgent(alpha=learning_rate, gamma=discount, exploration_strategy=exploration,
                         algo=agent_strategy, dim_a=3, eps_0=epsilon, tau=tau)

    env = make('rps')
    global_step = 0
    for ep in tqdm(range(train_episodes)):
        # do an episode against a random opponent
        opponent = opponents[np.random.choice(len(opponents))]
        trainer = env.train([None, opponent])
        print(f'ep {ep} - opponent {opponent}')

        observation = trainer.reset()
        action_buffer = []
        for t in range(1000):
            # save last opponent action
            if observation.step > 0:
                action_buffer.append(observation.lastOpponentAction)

            if len(action_buffer) < history_length:
                # play random actions at the beginning
                action = np.random.randint(3)
            else:
                # learning step
                s = action_buffer[-history_length - 1:-1]
                a = action_buffer[-2]
                r = get_score(a, observation.lastOpponentAction)
                s_prime = action_buffer[-history_length:]
                agent.learn(s, a, r, s_prime)

                action = int(agent.act(s_prime))

            action_buffer.append(action)
            observation, reward, _, _ = trainer.step(action)

            if global_step % 10 == 0:
                writer.add_scalar('reward', reward, global_step)
            global_step += 1


if __name__ == '__main__':
    # only when this file is called directly
    train_agent()

else:
    alpha = 0.7
    discount = 0.3
    epsilon = 0.82
    epsilon_decay = 0.99999999
    history_length = 2
    agent_strategy = 'Q-learning'
    exploration = 'epsilon-greedy'
    # exploration = 'boltzmann'
    tau = 0.01

    action_buffer = []
    agent = TabularAgent(alpha=alpha, gamma=discount, exploration_strategy=exploration,
                         algo=agent_strategy, dim_a=3, eps_0=epsilon, eps_decay=epsilon_decay, tau=tau)


def act_func(observation, configuration=None):
    global agent
    global action_buffer

    # save last opponent action
    if observation.step > 0:
        action_buffer.append(observation.lastOpponentAction)

    if len(action_buffer) < history_length + 2:
        # play random actions at the beginning
        action = np.random.randint(3)
    else:
        # observe last actions
        s = action_buffer[-history_length - 2:-2]
        a = action_buffer[-2]
        r = get_score(a, observation.lastOpponentAction)
        s_prime = action_buffer[-history_length:]
        agent.learn(s, a, r, s_prime)
        action = int(agent.act(action_buffer[-history_length:]))

    action_buffer.append(action)
    return action
