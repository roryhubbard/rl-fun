import gym
import numpy as np
from policy_evaluation_FrozenLake_v0 import evaluate_policy
from utils import initialize_policy, get_state_action_value, get_greedy_action


SEED = 0
# 0=Left, 1=Down, 2=Right, 3=Up


def get_policy_action(s, policy):
    # return action with highest probability
    return policy[s].argmax()


def iterate_policy(policy, env, gamma, n_iter=100):
    """
    input:
    - policy: array[states X actions] with probabilities for each action
    - env: environment
    - gamma: discount_rate
    - n_iter: max # of times to iterate policy
    output:
    - policy: optimal policy
    """
    for k in range(n_iter):
        policy_stable = True
        V = evaluate_policy(policy, env, gamma)
        for s in range(env.nS):
            old_a = get_policy_action(s, policy)
            greedy_a, _ = get_greedy_action(s, env, V, gamma)
            policy[s] = np.eye(env.nA)[greedy_a]
            if old_a != greedy_a:
                policy_stable = False

        if policy_stable:
            print(f'policy iteration stabilized after {k+1} iterations')
            break
        elif k == n_iter - 1:
            print('policy iteration never stabilized')

    return policy, V


def main():
    env = gym.make('FrozenLake-v0')
    env.seed(SEED)
    env.reset()
    env.render()

    policy = initialize_policy(env.nA, env.nS)
    print('initial policy')
    print(policy)

    # set discount-rate [0, 1): becomes unstable if set to 1!
    gamma = .9

    optimal_policy, V = iterate_policy(policy, env, gamma)

    print('optimal grid policy: 0=Left, 1=Down, 2=Right, 3=Up')
    print(optimal_policy.argmax(axis=1)
          .reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))))

    print('optimal state value function')
    print(V.reshape(int(np.sqrt(env.nS)), int(np.sqrt(env.nS))))

    env.close()


if __name__ == "__main__":
    main()
