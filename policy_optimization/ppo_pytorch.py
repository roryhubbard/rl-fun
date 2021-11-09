from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from models_pytorch import Actor, Critic
from core import compute_critic_loss, compute_actor_loss, \
  get_advantages_and_returns, rollout, moving_average


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

