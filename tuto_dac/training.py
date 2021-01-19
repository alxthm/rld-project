import pickle

import sys

try:
    """ For kaggle environment """
    sys.path.append("/kaggle_simulations/agent")
except:
    pass

from agents.randomagent import RandomAgent

agent = RandomAgent()
### Do the training

with open("agents/randomagent.pkl", "wb") as f:
    pickle.dump(agent, f)
