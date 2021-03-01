import numpy as np
import torch
from torch.distributions import Categorical


def act_levi(observation, model, my_last_action, last_h):
    state = np.zeros(7)
    if observation.step == 0:
        last_h = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
    else:
        # one-hot encode my last action and opponent action as well as t
        state = np.zeros(7)
        state[my_last_action] = 1
        state[observation.lastOpponentAction] = 1
        state[-1] = observation.step

    with torch.no_grad():
        logits, h = model.pi(torch.from_numpy(state).float(), last_h)
    logits = logits.view(-1)
    m = Categorical(logits=logits)
    a = m.sample().item()
    return a, h
