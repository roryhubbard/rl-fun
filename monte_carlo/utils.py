import random
from collections import defaultdict
import numpy as np


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


def get_random_policy(env):
    return defaultdict(
        lambda: np.ones(env.action_space.n) / env.action_space.n)


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


def get_episode_epsilon_greedy(Q, env, epsilon):
    """
    episode = [S0, A0, R1, S1, A1, R2, ...]
    """
    observation = env.reset()
    episode = []
    done = False

    while not done:
        if random.uniform(0.0, 1.0) < epsilon:
            action = random.choice(tuple(range(env.action_space.n)))
        else:
            action = get_greedy_action(Q, observation)

        next_observation, reward, done, _ = env.step(action)
        episode.append((observation, action, reward))
        observation = next_observation

    return episode


def get_greedy_action(Q, observation):
    return Q[observation].argmax()


def get_disounted_reward(rewards, discount_rate):
    """
    G(t) = R(t+1) + R(t+2) * gamma + ... + R(T) * gamma**(T-1)
    """
    discounted_reward = 0
    for i, r in enumerate(rewards):
        discounted_reward += r * discount_rate**i

    return discounted_reward
