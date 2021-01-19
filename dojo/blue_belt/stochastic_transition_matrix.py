import numpy as np
import json
#import torch

matrix = np.ones((3,3,3)) * (1/3) #so we can choose object based on what we chose and what the opponent chose transition matrix
matrix_freq = np.ones((3,3,3)) #frequency matrix
prev_me = 0
prev_op = 0
#print(state_dict)

def copy_opponent_agent (observation, configuration):
    
    global prev_me, prev_op, matrix, matrix_freq
        
    if observation.step > 0:
        #return (observation.lastOpponentAction + 1)%3
        #prev_op = observation.lastOpponentAction #we store the last action of the opponent
        
        #from step > 1 we can update matrix because we know what we chose and what it chose
        if observation.step > 1:
            matrix_freq[prev_op, prev_me, observation.lastOpponentAction] += 1
            matrix[prev_op, prev_me, :] = matrix_freq[prev_op, prev_me, :] / np.sum(matrix_freq[prev_op, prev_me, :]) 
            
        
        prev_op = observation.lastOpponentAction #we store the last action of the opponent  
        
        #choose the optimal choice based on the transition matrix
        #choosing stochastically
        prev_me = (np.random.choice(3, p=matrix[prev_op, prev_me, :]) + 1) % 3
        
        #print(matrix) 
        
        state_dict = {"transition tensor" : matrix.tolist()}
        with open('transition_matrix.json', 'a') as outfile:
            json.dump(state_dict, outfile)
            outfile.write("\n")
        
        return prev_me
        
              
    else:
        #prev_me = np.random.randint(0,3)
        state_dict = {"transition tensor" : matrix.tolist()}
        with open('transition_matrix.json', 'w') as outfile:
            json.dump(state_dict, outfile)
            outfile.write("\n")
        prev_me = (np.random.choice(3, p=matrix[prev_op, prev_me, :]) + 1)%3
        return prev_me
        #json.dump(matrix_freq, outfile)
