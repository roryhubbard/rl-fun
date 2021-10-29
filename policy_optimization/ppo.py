from tqdm import tqdm
import gym
import numpy as np
from models import Actor, Critic


def main():
  # states: [x, theta, x', theta']
  env = gym.make('InvertedPendulum-v2')

  T = 2048 # environement steps per update
  batch_size = 64
  epochs = 10
  discount = 0.99
  clipping_epsilon = 0.2
  lam = 0.95 # GAE parameter
  total_timesteps = 1000000

  actor = Actor()
  critic = Critic()

  n_updates = total_timesteps // T
  n_batches_per_update = T // batch_size

  for update in range(n_updates):

    states = []
    actions = []
    rewards = []
    dones = []
    state_values = []
    advantages = []
    done = True

    for _ in range(T):
      if done:
        state = env.reset()
        episode_lenth = 0

      states.append(state)
      state_value = critic.get_value(state)
      state_values.append(state_value)

      #action = env.action_space.sample()
      action = actor.get_action(state)
      next_state, reward, done, _ = env.step(action)

      if not done:
        state = next_state
      actions.append(action)
      rewards.append(reward)
      dones.append(done)

      episode_length += 1

      for i in range(episode_length):
        delta = rewards[i] + discount * critic.get_value(state[i+1]) - critic.get_value[i]
        advantage_estimate += (discount * lam)**t * delta

      #env.render()

    advantage_estimates = []
    value_estimates = []


    for k in range(epochs):
      for n in range(0, n_batches_per_update, batch_size):
        pass

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

