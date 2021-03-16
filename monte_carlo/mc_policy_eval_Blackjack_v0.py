import gym
import numpy as np
from collections import defaultdict
from plotting import plot_value_function


SEED = 0


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


def get_episode(policy, env):
    """
    episode = [S0, A0, R1, S1, A1, R2, ...]
    """
    observation = env.reset()
    episode = []
    done = False
    while not done:
        action = policy(observation)
        next_observation, reward, done, _ = env.step(action)
        episode.append((observation, action, reward))
        observation = next_observation
    return episode


def get_disounted_reward(rewards, discount_rate):
    """
    G(t) = R(t+1) + R(t+2) * gamma + ... + R(T) * gamma**(T-1)
    """
    discounted_reward = 0
    for i, r in enumerate(rewards):
        discounted_reward += r * discount_rate**i
    return discounted_reward


def mc_policy_eval(policy, env, num_episodes=10000, discount_rate=1.0):
    """
    First visit monte carlo policy evaluation
    input:
    - policy: map observation to action
    - env: environment
    - num_episodes: # of episodes to run
    - discount_rate: gamma
    output:
    - V: state value function according to policy
    """
    state_counter = defaultdict(int)
    V = defaultdict(float)
    for t in range(num_episodes):
        episode = get_episode(policy, env)
        visited_states = set()
        for i, (state, _action, _reward) in enumerate(episode):
            # only allow first visit to contribute to state value
            if state not in visited_states:
                visited_states.add(state)
                state_counter[state] += 1
                G = get_disounted_reward(
                    list(map(lambda x: x[2], episode[i:])), discount_rate)
                # incremental mean
                V[state] += (G - V[state]) / state_counter[state]
    return V


def main():
    env = gym.make('Blackjack-v0')
    env.seed(SEED)

    V = mc_policy_eval(sample_policy, env, 10000)
    plot_value_function(V, title="10,000 Episodes")

    V = mc_policy_eval(sample_policy, env, 500000)
    plot_value_function(V, title="500,000 Episodes")

    env.close()


if __name__ == "__main__":
    main()
