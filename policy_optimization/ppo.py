from tqdm import tqdm
import gym
import numpy as np
from models import Actor, Critic


def main():
  # states: [x, theta, x', theta']
  env = gym.make('InvertedPendulum-v2')

  T = 2048
  epochs = 10
  batch_size = 64
  discount = 0.99
  clipping_epsilon = 0.2
  lam = 0.95 # GAE parameter
  total_timesteps = 1000000

  actor = Actor()
  critic = Critic()

  D = [] # experience replay bufffer [(state, action, reward, next_state), ...]

  for t in range(T):
    state = env.reset()
    done = False

    while not done:
      #action = env.action_space.sample()
      state_value = critic.get_value(state)
      action = actor.get_action(state)

      next_state, reward, done, _ = env.step(action)

      experience = (state, action, reward, next_state) if not done \
          else (state, action, reward, None)

      D.append(experience)
      state = next_state

      current_advantage = reward + discount * critic.get_value(next_state) - state_value \
          if not done else reward - state_value

      advantage_estimate += (discount * lam)**t * current_advantage

      env.render()

    for k in range(epochs):
      pass

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

