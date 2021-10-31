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
      state_value = critic.forward(state)
      action, log_prob = actor.forward(state)
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

    # rewards to go are the discounted rewards through an episode which can be recovered
    # by adding the state values back to the advantages (look at delta calculation above)
    # this does make the rewards discounted by discount * lam instead of just discount
    rewards_to_go = advantages + state_values

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

