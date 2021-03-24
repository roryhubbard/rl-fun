import random
import gym
import numpy as np


SEED = 0


def initialize_state_action_values(n_states, n_actions, random=False):
    """
    Input:
    - n_states: number of states in environment
    - n_actions: number of actions in environment
    - random: boolean that decides to initialize state-action values randomly
      or make them all equal

    Output:
    - Q: state-action value function
    """
    if random:
        Q = np.random.default_rng(SEED).dirichlet(np.ones(n_actions),
                                                  size=n_states)
    else:
        # all state-action values are equal
        Q = np.ones((n_states, n_actions)) / n_actions
    return Q


def get_greedy_action(Q, observation):
    return Q[observation].argmax()


def sarsa(env, num_episodes, discount_rate=1.0, learning_rate=0.5, epsilon=0.2):
    """
    Estimate optimal state-value function Q using SARSA.

    Input:
    - env: environment
    - num_episodes: # of episodes to run
    - discount_rate (gamma): how much to discount future rewards
    - learning_rate (alpha): how much to update Q based on a given TD-error
    - epsilon: probability of "exploring" or choosing a random action

    Output:
    - Q: optimal state-value function
    """
    Q = initialize_state_action_values(env.nS, env.nA)

    for t in range(num_episodes):
        state = env.reset()
        action = get_greedy_action(Q, state)
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)

            if random.uniform(0.0, 1.0) < epsilon:
                next_action = random.choice(tuple(range(env.nA)))
            else:
                next_action = get_greedy_action(Q, state)

            Q[state][action] += learning_rate * (
                reward + discount_rate * Q[next_state][next_action] -
                Q[state][action])

            state = next_state
            action = next_action
     
    return Q


def main():
    env = gym.make('CliffWalking-v0')
    env.seed(SEED)
    Q = sarsa(env, 10)


if __name__ == "__main__":
    main()
