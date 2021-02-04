from kaggle_environments import make

env = make('rps', configuration={'episodeSteps': 10}, debug=True)
steps = env.run(['dojo/my_little_dojo/saitama_tabular.py', 'statistical'])
for s in steps:
    print(s)
