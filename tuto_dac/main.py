""" To make the submission, tar -cvzf soumission.tar.gz agents/ main.py """

import pickle
import sys
import random
import logging

try:
    """ For kaggle environment, add to module path the local directory """
    sys.path.append("/kaggle_simulations/agent")
except:
    pass

from agents.randomagent import RandomAgent

## Load the local model
try:
    with open("agents/randomagent.pkl", "rb") as f:
        model = pickle.load(f)
except:
    """ If played in kaggle environment """
    with open("/kaggle_simulations/agent/agents/randomagent.pkl", "rb") as f:
        model = pickle.load(f)

import time


def random_agent(obs, config):
    if obs.step == 0:
        return random.randint(0, 2)
    return model.play(obs.lastOpponentAction, config)
