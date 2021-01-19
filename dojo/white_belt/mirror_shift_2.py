def mirror_shift_opponent_agent_2(observation, configuration):
  if observation.step > 0:
    return (observation.lastOpponentAction + 2) % 3
  else:
    return 0
