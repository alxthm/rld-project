# Playing Rock, Paper, Scissors with Reinforcement Learning
This is an attempt to apply RL algorithms at the [Rock, Paper, Scissors (RPS) Kaggle competition](https://www.kaggle.com/c/rock-paper-scissors/overview).

```
.
├── dojo
│     ├── black_belt
│     ├── blue_belt
│     ├── white_belt
│     └── my_little_dojo
├── runs
├── eval.py
├── ppo_model.py
├── ppo_train.py
└── test.py
```

## Algorithms
We implemented and evaluated our algorithms against a number of baselines from [1] (white, blue and black belt baselines).

Our agents are located in the `dojo/my_little_dojo` folder:
- **Simple Multi-armed-bandits**: `jotaro.py`
- **Bandit of bandits** (*Meta-bandits/Banditception*): `giorno.py`
- **Tabular Q-learning**: `saitama.py`
- **PPO-LSTM** (considering a POMDP setting): `levi.py`

_Note:_ Unlike other agents, the PPO-LSTM agent does not learn a strategy online from scratch, but is pre-trained against a number of white belt baselines offline. Weights of pre-trained agents are located in the `runs` folder.

## How to run
Our implementation uses PyTorch and, in addition to common python data libraries, you may need to install the `kaggle-environments` library.

To evaluate an agent against the other baselines, run the `eval.py` specifying the agent name and the baseline against which you want to evaluate it.

I you want to run an algorithm in debug mode (not possible with the default Kaggle environment), you can use the `test.py` file, which reproduces the RPS game. This can be useful if `eval.py` fails silently (agent returning None), but please make sure you have all required libraries installed (`torch`, `tensorboard`, etc).

Run `ppo_train.py` if you want (specifying one or multiple opponents) to train the PPO-LSTM agent.

## References
[1] RPS Dojo _Kaggle Notebook_ (https://www.kaggle.com/chankhavu/rps-dojo)
