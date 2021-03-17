import gym
import numpy as np
from collections import defaultdict
from utils import get_disounted_reward, \
    get_episode_epsilon_greedy_from_policy, get_greedy_action, sample_policy
from plotting import plot_value_function


SEED = 0


def mc_control_importanc_sampling(env, num_episodes,
                              discount_rate=1.0, epsilon=0.2):
    """
    Importance sampling, off-policy monte carlo control
    input:
    - env: environment
    - num_episodes: # of episodes to run
    - discount_rate: gamma
    - epsilon: probability of "exploring" or choosing a random action
    output:
    - policy: epsilon greedy optimal policy
    - Q: state-action value function
    """
    b = sample_policy
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = defaultdict(int)

    for t in range(num_episodes):
        episode = get_episode_epsilon_greedy_from_policy(b, env, epsilon)
        W = 1

        for i, (state, action, _reward) in enumerate(episode):
            G = get_disounted_reward(
                list(map(lambda x: x[2], episode[i:])), discount_rate)
            C[state][action] += W

            Q[state][action] += W / C * (G - Q[state][action])
            policy[state] = get_greedy_action(Q, state)

            if action != policy[state]:
                break

            W /= _____________________

    return policy, Q


def main():
    env = gym.make('Blackjack-v0')
    env.seed(SEED)

    policy, Q = mc_control_epsilon_greedy(env, 500000)

    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plot_value_function(V, title="Optimal Value Function")

    env.close()


if __name__ == "__main__":
    main()
