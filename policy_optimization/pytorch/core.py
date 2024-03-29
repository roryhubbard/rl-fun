import numpy as np
import torch


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

