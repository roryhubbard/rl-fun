from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from models_pytorch import Actor, Critic


def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w


def rollout(env, policy_net, value_net, horizon, nstates, max_ep_length):
  states = np.empty((horizon, nstates))
  actions = np.empty(horizon)
  rewards = np.empty(horizon)
  dones = np.empty(horizon)
  state_values = np.empty(horizon+1)
  log_probs = np.empty(horizon)

  state = env.reset()
  ep_rewards = []
  ep_reward = 0
  ep_length = 0

  for t in range(horizon):
    with torch.no_grad():
      state_tensor = torch.as_tensor(state, dtype=torch.float32)
      state_value = value_net(state_tensor)
      action, log_prob = policy_net(state_tensor)

    next_state, reward, done, _ = env.step(action)

    states[t] = state
    actions[t] = action
    rewards[t] = reward
    dones[t] = done
    state_values[t] = state_value
    log_probs[t] = log_prob
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
  state_values[horizon] = value_net(state_tensor) if not done else 0

  return states, actions, rewards, dones, state_values, log_probs, ep_rewards


def get_advantages_and_returns(dones, rewards, values, discount, lam, horizon):
  next_non_terminal = 1 - dones
  deltas = rewards + next_non_terminal * discount * values[1:] - values[:-1]
  gae = 0
  discounted_reward = values[horizon] # initial value is very last element of values
  advantages = np.empty(horizon)
  returns = np.empty(horizon) # discounted rewards

  for i in reversed(range(horizon)):
    gae = deltas[i] + next_non_terminal[i] * discount * lam * gae
    advantages[i] = gae
    discounted_reward = rewards[i] + next_non_terminal[i] * discount * discounted_reward
    returns[i] = discounted_reward

  advantages = (advantages - advantages.mean()) / advantages.std()

  return advantages, returns


def compute_actor_loss(actor, state, action, log_prob, advantage, clip_epsilon):
  _, current_log_probs = actor(state, action)
  ratios = torch.exp(current_log_probs - log_prob)
  clipped_ratios = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon)
  unclipped_surrogate = ratios * advantage
  clipped_surrogate = clipped_ratios * advantage
  return -(torch.min(unclipped_surrogate, clipped_surrogate)).mean()


def compute_critic_loss(critic, state, G):
  return ((critic(state) - G)**2).mean()

def main():
  env = gym.make('InvertedPendulum-v2')
  #env = gym.make('InvertedDoublePendulum-v2')
  # states: [x, theta, x', theta']
  # action: [horizontal force]
  nstates = env.observation_space.shape[0]
  nactions = env.action_space.shape[0]

  T = 2048 # environement steps per update
  batch_size = 64
  epochs = 10
  lr = 3e-4
  discount = 0.99
  clip_epsilon = 0.2
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

  episode_rewards = []
  critic_losses = []
  for update in tqdm(range(n_updates)):
    states, actions, rewards, dones, values, log_probs, ep_rewards = rollout(
      env, actor, critic, T, nstates, max_ep_length)
    episode_rewards += ep_rewards

    advantages, returns = get_advantages_and_returns(dones, rewards, values, discount, lam, T)

    idx = np.arange(T)

    states = torch.as_tensor(states, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)

    for k in range(epochs):
      np.random.default_rng().shuffle(idx)

      for n in range(0, n_batches_per_update, batch_size):
        batch_idx = idx[n:n+batch_size]
        state = states[batch_idx]
        action = actions[batch_idx]
        log_prob = log_probs[batch_idx]
        advantage = advantages[batch_idx]
        G = returns[batch_idx]

        actor_loss = compute_actor_loss(actor, state, action,
                                        log_prob, advantage, clip_epsilon)
        critic_loss = compute_critic_loss(critic, state, G)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        critic_losses.append(critic_loss.item())

  env.close()

  torch.save(actor, 'actor.pt')

  fig, ax = plt.subplots()
  ax.plot(moving_average(episode_rewards, 100))
  plt.show()
  plt.close()

  fig, ax = plt.subplots()
  ax.plot(moving_average(critic_losses, 10))
  plt.show()
  plt.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

