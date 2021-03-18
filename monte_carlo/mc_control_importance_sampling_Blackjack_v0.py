import gym
import numpy as np
from collections import defaultdict
from utils import get_disounted_reward, \
    get_episode_epsilon_greedy, get_greedy_action, get_random_policy
from plotting import plot_value_function


SEED = 0


def mc_control_importance_sampling(env, num_episodes,
                              discount_rate=1.0, epsilon=0.2):
    """
    Importance sampling, off-policy monte carlo control
    input:
    - env: environment
    - num_episodes: # of episodes to run
    - discount_rate: gamma
    - epsilon: probability of "exploring" or choosing a random action
    output:
    - target_policy: epsilon greedy optimal policy
    - Q: state-action value function
    """
    b = get_random_policy(env)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = defaultdict(int)  # deterministic

    for t in range(num_episodes):
        # generate episode with behavior policy (b)
        episode = get_episode_epsilon_greedy(b, env, epsilon)
        W = 1

        # loop backwards through episode
        for i, (state, action, _reward) in enumerate(list(reversed(episode))):
            G = get_disounted_reward(
                list(map(lambda x: x[2], episode[i:])), discount_rate)

            # add to running some of importance-sampling ratios
            # for this state-action pair
            C[state][action] += W

            # update state-action value
            Q[state][action] += W / C[state][action] * (G - Q[state][action])
            # update target policy with updated Q value
            target_policy[state] = get_greedy_action(Q, state)

            # if the action taken by the behavior policy is not the same as
            # what the target policy would take, the importance-sampling
            # ration becomes 0 because target policy is deterministic, so break
            if action != target_policy[state].argmax():
                break

            # update importance sampling ratio, numerator is 1 because target
            # policy is deterministic
            W /= b[state][action]

    return target_policy, Q


def main():
    env = gym.make('Blackjack-v0')
    env.seed(SEED)

    policy, Q = mc_control_importance_sampling(env, 500000)

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
