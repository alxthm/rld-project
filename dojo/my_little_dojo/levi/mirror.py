import torch

from dojo.my_little_dojo.levi import act_levi
from ppo_model import PPO

model_path = 'runs/levi/20210228_011059/model_white_belt/mirror.pth'
model = PPO()
model.eval()
model.load_state_dict(torch.load(model_path))
my_last_action = None
last_h = None


def act_levi_(observation, configuration):
    global model, my_last_action, last_h
    a, last_h = act_levi(observation, model, my_last_action, last_h)
    my_last_action = a
    return a
