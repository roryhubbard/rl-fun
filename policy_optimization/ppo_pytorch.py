from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from models_pytorch import Actor, Critic


def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w


def discount_cumsum(x, discount):
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def main():
  env = gym.make('InvertedPendulum-v2')
  # states: [x, theta, x', theta']
  # action: [horizontal force]
  nstates = 4
  nactions = 1

  T = 2048 # environement steps per update
  batch_size = 64
  epochs = 10
  lr = 3e-4
  discount = 0.99
  clipping_epsilon = 0.2
  lam = 0.95 # GAE parameter
  total_timesteps = 1000000
  max_ep_length = 1000

  actor = Actor(nstates, nactions)
  critic = Critic(nstates)

  actor_optimizer = Adam(actor.parameters(), lr=lr)
  critic_optimizer = Adam(critic.parameters(), lr=lr)

  n_updates = total_timesteps // T
  if total_timesteps % T != 0:
    n_updates += 1

  n_batches_per_update = T // batch_size
  if T % batch_size != 0:
    n_batches_per_update += 1

  ep_rewards = []
  for update in tqdm(range(n_updates)):
    states = np.empty((T, nstates))
    actions = np.empty(T)
    rewards = np.empty(T)
    returns = np.empty(T) # discounted rewards
    dones = np.empty(T)
    state_values = np.empty(T+1)
    log_probs = np.empty(T)
    advantages = np.empty(T)

    state = env.reset()
    ep_reward = 0
    ep_length = 0

    for t in range(T):
      with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        state_value = critic(state_tensor)
        action, log_prob = actor(state_tensor)

      next_state, reward, done, _ = env.step(action)

      states[t] = state
      actions[t] = action.item()
      rewards[t] = reward
      dones[t] = done
      state_values[t] = state_value.item()
      log_probs[t] = log_prob.item()
      ep_reward += reward
      ep_length += 1

      if done or ep_length == max_ep_length:
        state = env.reset()
        ep_rewards.append(ep_reward)
        ep_reward = 0
        ep_length = 0
      else:
        state = next_state

    # compute state value estimate and reward for very last state
    # state value is 0 if it ended in a terminal state (done==true)
    # bootstrap the initial discounted reward to this state value
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    state_values[T] = critic(state_tensor) if not done else 0
    discounted_reward = state_values[T]

    next_non_terminal = 1 - dones
    deltas = rewards + next_non_terminal * discount * state_values[1:] - state_values[:-1]
    gae = 0
    for i in reversed(range(T)):
      gae = deltas[i] + next_non_terminal[i] * discount * lam * gae
      advantages[i] = gae
      discounted_reward = rewards[i] + next_non_terminal[i] * discount * discounted_reward
      returns[i] = discounted_reward

    advantages = (advantages - advantages.mean()) / advantages.std()
    idx = np.arange(T)
    critic_losses = []
    for k in range(epochs):
      np.random.default_rng().shuffle(idx)

      for n in range(0, n_batches_per_update, batch_size):
        batch_idx = idx[n:n+batch_size]
        batch_states = states[batch_idx]
        batch_actions = actions[batch_idx]
        batch_returns = returns[batch_idx]
        batch_state_values = state_values[batch_idx]
        batch_A = advantages[batch_idx]
        batch_log_probs = log_probs[batch_idx]

        batch_state_tensor = torch.as_tensor(batch_states, dtype=torch.float32)
        batch_action_tensor = torch.as_tensor(batch_actions, dtype=torch.float32)
        batch_A_tensor = torch.as_tensor(batch_A, dtype=torch.float32)
        batch_log_prob_tensor = torch.as_tensor(batch_log_probs, dtype=torch.float32)
        batch_return_tensor = torch.as_tensor(batch_returns, dtype=torch.float32)

        _, current_log_probs = actor(batch_state_tensor, batch_action_tensor)
        ratios = torch.exp(current_log_probs - batch_log_prob_tensor)
        clipped_ratios = torch.clamp(ratios, 1-clipping_epsilon, 1+clipping_epsilon)
        unclipped_surrogate = ratios * batch_A_tensor
        clipped_surrogate = clipped_ratios * batch_A_tensor
        actor_loss = -(torch.min(unclipped_surrogate, clipped_surrogate)).mean()

        current_state_values = critic(batch_state_tensor)
        critic_loss = ((current_state_values - batch_return_tensor)**2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        critic_losses.append(critic_loss.item())

  fig, ax = plt.subplots()
  ax.plot(moving_average(ep_rewards, 500))
  ax.plot(critic_losses)
  plt.show()
  plt.close()

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

