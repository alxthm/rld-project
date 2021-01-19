def mirror_opponent_agent(observation, configuration):
  if observation.step > 0:
    return observation.lastOpponentAction
  else:
    return 0
