def mirror_shift_opponent_agent_1(observation, configuration):
  if observation.step > 0:
    return (observation.lastOpponentAction + 1) % 3
  else:
    return 0
