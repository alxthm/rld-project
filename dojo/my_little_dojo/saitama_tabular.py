import math
from collections import defaultdict
from typing import List

import numpy as np

discount = 0.9
history_length = 5

agent_strategy = 'Q-learning'
learning_rate = 0.1
exploration = 'epsilon-greedy'
epsilon = 0.5
# exploration = 'boltzmann'
tau = 0.01


# agent = 'SARSA'

def get_score(left_move: int, right_move: int):
    delta = (
        right_move - left_move
        if (left_move + right_move) % 2 == 0
        else left_move - right_move
    )
    return 0 if delta == 0 else math.copysign(1, delta)


class TabularAgent:
    def __init__(self, alpha, gamma, exploration_strategy, algo, dim_a, eps_0, tau):
        self.Q = defaultdict(lambda: np.zeros(dim_a))
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.algorithm = algo
        self.eps_0 = eps_0
        self.episode = 0
        self.tau = tau
        self.dim_a = dim_a

    def act(self, obs):
        obs = str(obs)
        if self.exploration_strategy == 'epsilon-greedy':
            eps = self.eps_0 / (1 + 0.01 * self.episode)
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


agent = TabularAgent(alpha=learning_rate, gamma=discount, exploration_strategy=exploration, algo=agent_strategy,
                     dim_a=3, eps_0=epsilon, tau=tau)
action_buffer = []


def play(observation, configuration):
    global agent

    # save last opponent action
    if observation.lastOpponentAction is not None:
        action_buffer.append(observation.lastOpponentAction)

    # play random actions at the beginning
    if len(action_buffer) < history_length:
        action = np.random.randint(3)

    # learning step, then act
    else:
        s = action_buffer[-history_length - 1:-1]
        a = action_buffer[-2]
        r = observation.reward
        assert get_score(action_buffer[-1], action_buffer[-2]) == r
        s_prime = action_buffer[-history_length:]

        agent.learn(s, a, r, s_prime)
        action = agent.act(s_prime)

    if len(action_buffer) > 2 * (configuration.episodeSteps - 2):
        agent.episode += 1
        print(f'len buffer {len(action_buffer)} - increasing ep to {agent.episode}')

    action_buffer.append(action)
    return action
