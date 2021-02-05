import random

import numpy as np
import torch
import torch.nn.functional as F

action_history = []
opponent_history = []
reward_state = 0
reward_history = []
context_history = []
context_size = 12
A_list = [np.eye(context_size*3) for _ in range(3)]
b_list = [np.zeros((context_size*3, 1)) for _ in range(3)]
alpha = 0.15
probas = [0, 0, 0]


def get_context(actions_self, opponent_actions, ctx_size):
    context = []
    for i in range(0, ctx_size // 2):
        context.append(opponent_actions[len(opponent_actions) - i - 1])
        context.append(actions_self[len(actions_self) - i - 1])
    context = F.one_hot(torch.tensor(context), num_classes=3)
    context = context.numpy().flatten().reshape(context_size*3, 1)  # make it a vector
    return context


def actions_bandit(observation, configuration):
    global action_history
    global opponent_history
    global reward_history
    global context_size
    global reward_state
    global alpha
    global context_history

    if observation.step > 0:
        opponent_history.append(observation.lastOpponentAction)
        reward_history.append(observation.reward - reward_state)  # register the reward of the last action
        reward_state = observation.reward  # update reward status

    if observation.step >= context_size:
        # update A and b for LinUCB
        if len(context_history) > 0:
            x_t = context_history[-1]
            last_action = action_history[-1]
            A_list[last_action] += x_t @ x_t.T
            b_list[last_action] += reward_history[-1] * x_t

        # LinUCB estimation
        x_t = get_context(action_history, opponent_history, context_size)

        for i in range(3):
            A = A_list[i]
            b = b_list[i]
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            proba_i = theta.T @ x_t + alpha * np.sqrt(x_t.T @ A_inv @ x_t)
            probas[i] = proba_i[0][0]

        print(probas)
        choice = int(np.argmax(probas))
        context_history.append(x_t)
    else:
        choice = random.randint(0, 2)
    action_history.append(choice)  # register action
    # print(reward_history)
    return choice
