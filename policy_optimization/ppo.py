from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
from models import Actor, Critic
from core import get_advantages_and_returns, rollout, moving_average


def main():
  env = gym.make('InvertedPendulum-v2')
  # states: [x, theta, x', theta']
  # action: [horizontal force]
  nstates=4
  nactions=1

  T = 2048 # environement steps per update
  batch_size = 64
  epochs = 10
  lr = 0.01
  discount = 0.99
  clipping_epsilon = 0.2
  lam = 0.95 # GAE parameter
  total_timesteps = 1000000

  actor = Actor(nstates, nactions)
  critic = Critic(nstates)

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

    for k in range(epochs):
      np.random.default_rng().shuffle(idx)

      for n in range(0, n_batches_per_update, batch_size):
        batch_idx = idx[n:n+batch_size]
        state = states[batch_idx]
        action = actions[batch_idx]
        log_prob = log_probs[batch_idx]
        advantage = advantages[batch_idx]
        G = returns[batch_idx]

        _, current_log_probs = actor.forward(batch_states,
                                             batch_actions, requires_grad=True)
        ratios = np.exp(current_log_probs - batch_log_probs)
        clipped_ratios = np.minimum(1+clipping_epsilon,
                                    np.maximum(1-clipping_epsilon, ratios))

        unclipped_surrogate = ratios * batch_A
        clipped_surrogate = clipped_ratios * batch_A
        actor_loss = -np.minimum(unclipped_surrogate, clipped_surrogate).mean()

        current_state_values = critic.forward(batch_states, requires_grad=True)
        critic_loss = ((current_state_values - batch_returns)**2).mean()

        # derivative of actor_loss w.r.t current_log_probs
        dAL_dlp = -unclipped_surrogate
        # derivative of clipped_ratios w.r.t ratios
        dcr_dr = np.zeros_like(ratios)
        dcr_dr[(ratios < 1 + clipping_epsilon)
             & (ratios > 1 - clipping_epsilon)] = 1.0
        # only include the derivative of the clipped_ratio if the clipped_ratio was used
        clipped_used_idx = clipped_surrogate < unclipped_surrogate
        dAL_dlp[clipped_used_idx] *= dcr_dr[clipped_used_idx]

        # derivative of critic_loss w.r.t current_state_values
        dCL_dsv = current_state_values - batch_returns

        actor.backward(dAL_dlp)
        critic.backward(dCL_dsv)

        actor.optimization_step(lr)
        critic.optimization_step(lr)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

  env.close()

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

