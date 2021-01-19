import pandas as pd
import numpy as np
import json


# base class for all agents, random agent
class agent():
    def initial_step(self):
        return np.random.randint(3)
    
    def history_step(self, history):
        return np.random.randint(3)
    
    def step(self, history):
        if len(history) == 0:
            return int(self.initial_step())
        else:
            return int(self.history_step(history))
    
# agent that returns (previousCompetitorStep + shift) % 3
class mirror_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['competitorStep'] + self.shift) % 3
    
    
# agent that returns (previousPlayerStep + shift) % 3
class self_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['step'] + self.shift) % 3    


# agent that beats the most popular step of competitor
class popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['competitorStep'] for x in history])
        return (int(np.argmax(counts)) + 1) % 3

    
# agent that beats the agent that beats the most popular step of competitor
class anti_popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['step'] for x in history])
        return (int(np.argmax(counts)) + 2) % 3
    
    
# simple transition matrix: previous step -> next step
class transition_matrix(agent):
    def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type = 'step' 
        else:
            self.step_type = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        
    def history_step(self, history):
        matrix = np.zeros((3,3)) + self.init_value
        for i in range(len(history) - 1):
            matrix = (matrix - self.init_value) / self.decay + self.init_value
            matrix[int(history[i][self.step_type]), int(history[i+1][self.step_type])] += 1

        if  self.deterministic:
            step = np.argmax(matrix[int(history[-1][self.step_type])])
        else:
            step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type])]/matrix[int(history[-1][self.step_type])].sum())
        
        if self.counter_strategy:
            # we predict our step using transition matrix (as competitor can do) and beat probable competitor step
            return (step + 2) % 3 
        else:
            # we just predict competitors step and beat it
            return (step + 1) % 3
    

# similar to the transition matrix but rely on both previous steps
class transition_tensor(agent):
    
    def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type1 = 'step' 
            self.step_type2 = 'competitorStep'
        else:
            self.step_type2 = 'step' 
            self.step_type1 = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        
    def history_step(self, history):
        matrix = np.zeros((3,3, 3)) + 0.1
        for i in range(len(history) - 1):
            matrix = (matrix - self.init_value) / self.decay + self.init_value
            matrix[int(history[i][self.step_type1]), int(history[i][self.step_type2]), int(history[i+1][self.step_type1])] += 1

        if  self.deterministic:
            step = np.argmax(matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])])
        else:
            step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])]/matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])].sum())
        
        if self.counter_strategy:
            # we predict our step using transition matrix (as competitor can do) and beat probable competitor step
            return (step + 2) % 3 
        else:
            # we just predict competitors step and beat it
            return (step + 1) % 3

    
agents = {
    'mirror_0': mirror_shift(0),
    'mirror_1': mirror_shift(1),  
    'mirror_2': mirror_shift(2),
    'self_0': self_shift(0),
    'self_1': self_shift(1),  
    'self_2': self_shift(2),
    'popular_beater': popular_beater(),
    'anti_popular_beater': anti_popular_beater(),
    'random_transitison_matrix': transition_matrix(False, False),
    'determenistic_transitison_matrix': transition_matrix(True, False),
    'random_self_trans_matrix': transition_matrix(False, True),
    'determenistic_self_trans_matrix': transition_matrix(True, True),
    'random_transitison_tensor': transition_tensor(False, False),
    'determenistic_transitison_tensor': transition_tensor(True, False),
    'random_self_trans_tensor': transition_tensor(False, True),
    'determenistic_self_trans_tensor': transition_tensor(True, True),
    
    'random_transitison_matrix_decay': transition_matrix(False, False, decay = 1.1),
    'random_self_trans_matrix_decay': transition_matrix(False, True, decay = 1.1),
    'random_transitison_tensor_decay': transition_tensor(False, False, decay = 1.1),
    'random_self_trans_tensor_decay': transition_tensor(False, True, decay = 1.1),
    
    'determenistic_transitison_matrix_decay': transition_matrix(True, False, decay = 1.1),
    'determenistic_self_trans_matrix_decay': transition_matrix(True, True, decay = 1.1),
    'determenistic_transitison_tensor_decay': transition_tensor(True, False, decay = 1.1),
    'determenistic_self_trans_tensor_decay': transition_tensor(True, True, decay = 1.1),
    
    'random_transitison_matrix_decay2': transition_matrix(False, False, decay = 1.01),
    'random_self_trans_matrix_decay2': transition_matrix(False, True, decay = 1.01),
    'random_transitison_tensor_decay2': transition_tensor(False, False, decay = 1.01),
    'random_self_trans_tensor_decay2': transition_tensor(False, True, decay = 1.01),
    
    'determenistic_transitison_matrix_decay2': transition_matrix(True, False, decay = 1.01),
    'determenistic_self_trans_matrix_decay2': transition_matrix(True, True, decay = 1.01),
    'determenistic_transitison_tensor_decay2': transition_tensor(True, False, decay = 1.01),
    'determenistic_self_trans_tensor_decay2': transition_tensor(True, True, decay = 1.01),
}

    
def multi_armed_bandit_agent (observation, configuration):
    
    # bandits' params
    step_size = 3 # how much we increase a and b 
    decay_rate = 1.03 # how much do we decay old historical data
    
    # I don't see how to use any global variables, so will save everything to a CSV file
    # Using pandas for this is too much, but it can be useful later and it is convinient to analyze
    def save_history(history, file = 'history.csv'):
        pd.DataFrame(history).to_csv(file, index = False)

    def load_history(file = 'history.csv'):
        return pd.read_csv(file).to_dict('records')
    
    
    def log_step(step = None, history = None, agent = None, competitorStep = None):
        if step is None:
            step = np.random.randint(3)
        if history is None:
            history = []
        history.append({'step': step, 'competitorStep': competitorStep, 'agent': agent})
        save_history(history)
        return step
    
    def update_competitor_step(history, competitorStep):
        history[-1]['competitorStep'] = int(competitorStep)
        return history
        
    
    # load history
    if observation.step == 0:
        history = []
        bandit_state = {k:[1,1] for k in agents.keys()}
    else:
        history = update_competitor_step(load_history(), observation.lastOpponentAction)
        
        # load the state of the bandit
        with open('bandit.json') as json_file:
            bandit_state = json.load(json_file)
        
        # updating bandit_state using the result of the previous step
        # we can update all states even those that were not used
        for name, agent in agents.items():
            agent_step = agent.step(history[:-1])
            bandit_state[name][1] = (bandit_state[name][1] - 1) / decay_rate + 1
            bandit_state[name][0] = (bandit_state[name][0] - 1) / decay_rate + 1
            
            if (history[-1]['competitorStep'] - agent_step) % 3 == 1:
                bandit_state[name][1] += step_size
            elif (history[-1]['competitorStep'] - agent_step) % 3 == 2:
                bandit_state[name][0] += step_size
            else:
                bandit_state[name][0] += step_size/2
                bandit_state[name][1] += step_size/2
            
    with open('bandit.json', 'w') as outfile:
        json.dump(bandit_state, outfile)
    
    
    # generate random number from Beta distribution for each agent and select the most lucky one
    best_proba = -1
    best_agent = None
    for k in bandit_state.keys():
        proba = np.random.beta(bandit_state[k][0],bandit_state[k][1])
        if proba > best_proba:
            best_proba = proba
            best_agent = k
        
    step = agents[best_agent].step(history)
    
    return log_step(step, history, best_agent)
