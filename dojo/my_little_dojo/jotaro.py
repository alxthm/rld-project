import random

import numpy as np
import torch
import torch.nn.functional as F

action_history = []
opponent_history = []
reward_state = 0
reward_history = []
context_history = []
context_size = 8
A_list = [np.eye(context_size*3) for _ in range(3)]
b_list = [np.zeros(context_size*3) for _ in range(3)]
alpha = 0.15
probas = [0, 0, 0]


def get_context(actions_self, opponent_actions, ctx_size):
    context = []
    for i in range(0, ctx_size // 2):
        context.append(opponent_actions[len(opponent_actions) - i - 1])
        context.append(actions_self[len(actions_self) - i - 1])
    context = F.one_hot(torch.tensor(context), num_classes=3)
    return context.numpy().flatten()


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
            A_list[action_history[-1]] += context_history[-1].dot(context_history[-1])
            b_list[action_history[-1]] += reward_history[-1] * context_history[-1]

        # LinUCB estimation
        x_t = get_context(action_history, opponent_history, context_size)

        for i in range(3):
            A = A_list[i]
            b = b_list[i]
            A_inv = np.linalg.inv(A)
            theta = A_inv.dot(b)
            probas[i] = theta.dot(x_t) + alpha * np.sqrt(x_t.dot(A_inv).dot(x_t))

        # print(f'theta : {theta}')
        # print(f'second term : {np.sqrt(x_t.dot(A_inv).dot(x_t))}')
        print(probas)
        choice = int(np.argmax(probas))
        context_history.append(x_t)
    else:
        choice = random.randint(0, 2)
    action_history.append(choice)  # register action
    # print(reward_history)
    return choice
