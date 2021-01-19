import sys
import os
import kaggle_environments as kg

# Authorized libraries : Python Standard Library, gym, numpy, scipy, pytorch (cpu only)

import numpy as np
import torch

# Symbols :
# 0 : Rock
# 1 : Paper
# 2: Scissors

# Make the environment 
# * episodeSteps 	Maximum number of steps in the episode.
# * agentTimeout 	Maximum runtime(seconds) to initialize an agent.
# * actTimeout 	Maximum runtime(seconds) to obtain an action from an agent.
# * runTimeout 	Maximum runtime(seconds) of an episode(not necessarily DONE).


env = kg.make("rps", debug=True, configuration={
    "actTimeout": 1,
    "agentTimeout": 60,
    "runTimeout": 1200
})

# Agents from kaggle:  rock, paper, scissors, copy_opponent, reactionary, counter_reactionary, statistical
print(*env.agents)


# Agent definition : function of two variabless :
# *  observation:  dict : 
# **      'remainingOverageTime': int
# **      'lastOpponentAction': 0
# **      'step': int
# * configuration : 
# **      'episodeSteps': int
# **      'agentTimeout': int
# **      'actTimeout': int
# **      'runTimeout': int
# **      'isProduction': boolean
# **      'signs': int 
# **      'tieRewardThreshold': int

def other_agent(observation, configuration):
    return 0
def my_agent(observation, configuration):
    return 1




env.reset()
env.run([my_agent, my_agent])

# Evaluate 
agents = [my_agent, my_agent]
configuration = None
steps = 1000
num_episodes = 10
results = kg.evaluate('rps', agents, configuration, num_episodes=num_episodes)
print(results)

print('p')
