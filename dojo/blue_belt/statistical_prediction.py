
import random
import pydash
from collections import Counter

# Create a small amount of starting history
history = {
    "guess":      [0,1,2],
    "prediction": [0,1,2],
    "expected":   [0,1,2],
    "action":     [0,1,2],
    "opponent":   [0,1],
}
def statistical_prediction_agent(observation, configuration):    
    global history
    actions         = list(range(configuration.signs))  # [0,1,2]
    last_action     = history['action'][-1]
    opponent_action = observation.lastOpponentAction if observation.step > 0 else 2
    
    history['opponent'].append(opponent_action)

    # Make weighted random guess based on the complete move history, weighted towards relative moves based on our last action 
    move_frequency       = Counter(history['opponent'])
    response_frequency   = Counter(zip(history['action'], history['opponent'])) 
    move_weights         = [ move_frequency.get(n,1) + response_frequency.get((last_action,n),1) for n in range(configuration.signs) ] 
    guess                = random.choices( population=actions, weights=move_weights, k=1 )[0]
    
    # Compare our guess to how our opponent actually played
    guess_frequency      = Counter(zip(history['guess'], history['opponent']))
    guess_weights        = [ guess_frequency.get((guess,n),1) for n in range(configuration.signs) ]
    prediction           = random.choices( population=actions, weights=guess_weights, k=1 )[0]

    # Repeat, but based on how many times our prediction was correct
    prediction_frequency = Counter(zip(history['prediction'], history['opponent']))
    prediction_weights   = [ prediction_frequency.get((prediction,n),1) for n in range(configuration.signs) ]
    expected             = random.choices( population=actions, weights=prediction_weights, k=1 )[0]

    # Play the +1 counter move
    action = (expected + 1) % configuration.signs
    
    # Persist state
    history['guess'].append(guess)
    history['prediction'].append(prediction)
    history['expected'].append(expected)
    history['action'].append(action)

    # Print debug information
    print('opponent_action                = ', opponent_action)
    print('move_weights,       guess      = ', move_weights, guess)
    print('guess_weights,      prediction = ', guess_weights, prediction)
    print('prediction_weights, expected   = ', prediction_weights, expected)
    print('action                         = ', action)
    print()
    
    return action
