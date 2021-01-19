import re
import random
import pydash
from itertools import combinations_with_replacement

actions = list(combinations_with_replacement([2,1,0,2,1,0],3)) * 18
# random.shuffle(actions)
print('len(actions)',len(actions))
print(actions)
actions = pydash.flatten(actions)

# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def kaggle_agent(observation, configuration):    
    action = actions[observation.step] % configuration.signs
    return int(action)
