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
    log_probs = []
    done = True

    for _ in range(T):
      if done:
        state = env.reset()

      #action = env.action_space.sample()
      state_value = critic.get_value(state)
      action = actor.get_action(state)
      log_prob = np.exp(action)
      next_state, reward, done, _ = env.step(action)

      states.append(state)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)
      state_values.append(state_value)
      log_probs.append(log_prob)

      state = next_state

      #env.render()

    # compute state value estimate for very last state
    # so that the GAE of the last element in the state array can be computed
    state_values.append(critic.get_value(state))

    gae = 0
    advantages = [None] * T

    for i in reversed(range(T)):
      delta = rewards[i] + (1 - dones[i]) * discount * state_values[i+1] - state_values[i]
      gae = delta + (1 - dones[i]) * discount * lam * delta
      advantages[i] = gae

    for k in range(epochs):
      for n in range(0, n_batches_per_update, batch_size):
        pass

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

