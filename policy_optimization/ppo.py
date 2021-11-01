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
  n_batches_per_update = T // batch_size + 1
  if T % batch_size != 0:
    n_batches_per_update += 1

  for update in range(n_updates):
    states = np.empty(T)
    actions = np.empty(T)
    rewards = np.empty(T)
    returns = np.empty(T) # discounted
    dones = np.empty(T)
    state_values = np.empty(T+1)
    log_probs = np.empty(T)
    advantages = np.empty(T)

    state = env.reset()

    for t in range(T):
      state_value = critic.forward(state)
      action, log_prob = actor.forward(state)
      next_state, reward, done, _ = env.step(action)

      states[t] = state
      actions[t] = action
      rewards[t] = reward
      dones[t] = done
      state_values[t] = state_value
      log_probs[t] = log_prob

      state = env.reset() if done else next_state

    # compute state value estimate and reward for very last state
    # state value is 0 if it ended in a terminal state (done==true)
    # bootstrap the initial discounted reward to this state value
    state_values[T] = critic.get_value(state) if not done else 0
    discounted_reward = state_value[T]

    next_non_terminal = 1 - dones
    deltas = rewards + next_non_terminal * discount * state_values[1:] - state_values[:-1]
    gae = 0
    for i in reversed(range(len(deltas))):
      gae = delta[i] + next_non_terminal[i] * discount * lam * gae
      advantages[i] = gae
      discounted_reward = rewards[i] + next_non_terminal[i] * discount * discounted_reward
      returns[i] = discounted_reward

    rng = np.random.default_rng()
    idx = np.arange(T)
    for k in range(epochs):
      rng.shuffle(idx)
      for n in range(0, n_batches_per_update, batch_size):
        batch_idx = idx[n:n+batch_size]
        batch_states = states[batch_idx]
        batch_actions = actions[batch_idx]
        batch_rtg = rewards_to_go[batch_idx]
        batch_state_values = state_values[batch_idx]
        batch_advantages = advantages[batch_idx]
        batch_log_probs = log_probs[batch_idx]

        _, current_log_probs = actor.forward(batch_states, batch_actions, grad=True)

        ratios = np.exp(np.array(current_log_probs) - np.array(batch_log_probs))
        clipped_ratios = np.minimum(1+clipping_epsilon,
                                    np.maximum(1-clipping_epsilon, ratios))
        actor_loss = -np.minimum(ratios * batch_advantages,
                                 clipped_ratios * batch_advantages).mean()

        current_state_values = critic.forward(batch_states, grad=True)
        critic_loss = np.square(batch_rtg - current_state_values).mean()

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

