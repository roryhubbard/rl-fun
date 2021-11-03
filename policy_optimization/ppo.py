from tqdm import tqdm
import gym
import numpy as np
from models import Actor, Critic


def main():
  env = gym.make('InvertedPendulum-v2')
  # states: [x, theta, x', theta']
  # action: [horizontal force]
  nstates=4
  nactions=1

  T = 2048 # environement steps per update
  batch_size = 32
  epochs = 10
  lr = 0.01
  discount = 0.99
  clipping_epsilon = 0.2
  lam = 0.95 # GAE parameter
  total_timesteps = 1000000

  actor = Actor(nstates, nactions)
  critic = Critic(nstates)

  n_updates = total_timesteps // T
  n_batches_per_update = T // batch_size + 1
  if T % batch_size != 0:
    n_batches_per_update += 1

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
    state_values[T] = critic.forward(state) if not done else 0
    discounted_reward = state_values[T]

    next_non_terminal = 1 - dones
    deltas = rewards + next_non_terminal * discount * state_values[1:] - state_values[:-1]
    gae = 0
    for i in reversed(range(T)):
      gae = deltas[i] + next_non_terminal[i] * discount * lam * gae
      advantages[i] = gae
      discounted_reward = rewards[i] + next_non_terminal[i] * discount * discounted_reward
      returns[i] = discounted_reward

    idx = np.arange(T)
    actor_losses = []
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

        _, current_log_probs = actor.forward(batch_states,
                                             batch_actions, requires_grad=True)
        ratios = np.exp(current_log_probs - batch_log_probs)
        clipped_ratios = np.minimum(1+clipping_epsilon,
                                    np.maximum(1-clipping_epsilon, ratios))

        unclipped_surrogate = ratios * batch_A
        clipped_surrogate = clipped_ratios * batch_A
        actor_loss = -np.minimum(unclipped_surrogate, clipped_surrogate).mean()

        current_state_values = critic.forward(batch_states, requires_grad=True)
        critic_loss = 1/2 * np.square(batch_returns - current_state_values).mean()

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
        dCL_dsv = batch_returns - current_state_values

        actor.backward(dAL_dlp)
        critic.backward(dCL_dsv)

        actor.optimization_step(lr)
        critic.optimization_step(lr)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

