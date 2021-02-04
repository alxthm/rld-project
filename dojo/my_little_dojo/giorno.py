import random
import numpy as np
import torch
import torch.nn.functional as F


class Jotaro:
    def __init__(self, context_size, alpha):
        self.action_history = []
        self.opponent_history = []
        self.reward_state = 0
        self.reward_history = []
        self.context_history = []
        self.context_size = context_size
        self.A_list = [np.eye(context_size * 3) for _ in range(3)]
        self.b_list = [np.zeros((context_size * 3, 1)) for _ in range(3)]
        self.alpha = alpha
        self.probas = [0, 0, 0]

    def get_context(self, actions_self, opponent_actions):
        context = []
        for i in range(0, self.context_size // 2):
            context.append(opponent_actions[len(opponent_actions) - i - 1])
            context.append(actions_self[len(actions_self) - i - 1])
        context = F.one_hot(torch.tensor(context), num_classes=3)
        context = context.numpy().flatten().reshape(self.context_size * 3, 1)  # make it a vector
        return context

    def act(self, observation, configuration):

        if observation.step > 0:
            self.opponent_history.append(observation.lastOpponentAction)
            self.reward_history.append(observation.reward - self.reward_state)  # register the reward of the last action
            reward_state = observation.reward  # update reward status

        if observation.step >= self.context_size:
            # update A and b for LinUCB
            if len(self.context_history) > 0:
                x_t = self.context_history[-1]
                last_action = self.action_history[-1]
                self.A_list[last_action] += x_t @ x_t.T
                self.b_list[last_action] += self.reward_history[-1] * x_t

            # LinUCB estimation
            x_t = self.get_context(self.action_history, self.opponent_history)

            for i in range(3):
                A = self.A_list[i]
                b = self.b_list[i]
                A_inv = np.linalg.inv(A)
                theta = A_inv @ b
                proba_i = theta.T @ x_t + self.alpha * np.sqrt(x_t.T @ A_inv @ x_t)
                self.probas[i] = proba_i[0][0]

            choice = int(np.argmax(self.probas))
            self.context_history.append(x_t)
        else:
            choice = random.randint(0, 2)
        self.action_history.append(choice)  # register action
        # print(reward_history)
        return choice


# Global parameters for banditception

agents = [('agent-ctx_4-alpha-0.1', Jotaro(context_size=4, alpha=0.1)),
          ('agent-ctx_8-alpha-0.1', Jotaro(context_size=8, alpha=0.1)),
          ('agent-ctx_12-alpha-0.1', Jotaro(context_size=12, alpha=0.1)),
          ('agent-ctx_8-alpha-0.2', Jotaro(context_size=8, alpha=0.2)),
          ('agent-ctx_8-alpha-0.5', Jotaro(context_size=8, alpha=0.5))]

agent_choices_freq = [1, 1, 1, 1, 1]  # number of times agent i was chosen
reward_state = 0
reward_agents_history = [1, 1, 1, 1, 1]  # starts at a reward of 1 to prevent division by 0
agents_choice_history = []
reward_history = []


def banditception(observation, action):
    global agents
    global agent_choices_freq
    global reward_agents_history
    global reward_state
    global reward_history
    global agents_choice_history

    if observation.step > 0:
        # update of the rewards
        reward_history.append(observation.reward - reward_state)  # register the reward of the last action
        reward_state = observation.reward  # update reward status
        reward_agents_history[agents_choice_history[-1]] += reward_history[-1]  # update the rewards of the agents

    # action of the meta-bandit
    ub_estimate = [
        1 / agent_choices_freq[i] * reward_agents_history[i] + np.sqrt(
            2 * np.log(observation.step) / agent_choices_freq[i])
        for i in range(len(agents))]

    agent_chosen = int(np.argmax(ub_estimate))
    agent_choices_freq[agent_chosen] += 1
    agents_choice_history.append(agent_chosen)

    # actions of the sub-bandits
    _, agents_objects = zip(*agents)
    agents_actions = []

    for agent in agents_objects:
        agents_actions.append(agent.act(observation, action))

    return agents_actions[agent_chosen]
