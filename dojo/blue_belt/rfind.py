
import random
hist = []  # history of your moves
dict_last = {}
max_dict_key = 10
last_move = 0


def beat(x):
    return (x + 1) % 3


def predict():
    global dict_last
    global max_dict_key
    for i in reversed(range(min(len(hist), max_dict_key))):
        t = tuple(hist[-i:])
        if t in dict_last:
            return dict_last[t]
    return random.choice([0, 1, 2])


def update(move, op_move):
    global hist
    global dict_last
    global max_dict_key
    hist.append(move)
    for i in reversed(range(min(len(hist), max_dict_key))):
        t = tuple(hist[-i:])
        dict_last[t] = op_move


def run(observation, configuration):
    global last_move
    if observation.step == 0:
        last_move = random.choice([0, 1, 2])
        return last_move
    update(last_move, observation.lastOpponentAction)
    move = beat(predict())

    return move
