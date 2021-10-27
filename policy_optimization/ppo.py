import numpy as np
import gym
import torch


def main():
  # states: [x, theta, x', theta']
  env = gym.make('InvertedPendulum-v2')

  for episode in range(1): 
    obs = env.reset()
    done = False

    while not done:
      action = env.action_space.sample()
      nobs, reward, done, _ = env.step(action)
      env.render()

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

