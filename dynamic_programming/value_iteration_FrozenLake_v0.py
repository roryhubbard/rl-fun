import gym
import numpy as np
from utils import initialize_policy, get_state_action_value, get_greedy_action


SEED = 0
# 0=Left, 1=Down, 2=Right, 3=Up


def _get_policy_from_state_values(V, env, gamma):
    """
    input:
    - V: state value function
    - env: environment
    - gamma: discount_rate
    output:
    - policy driven by highest expected state values from V
    """
    policy = initialize_policy(env.nA, env.nS)
    for s in range(env.nS):
        greedy_a, _ = get_greedy_action(s, env, V, gamma)
        policy[s] = np.eye(env.nA)[greedy_a]
    return policy


def iterate_state_values(policy, env, gamma, n_iter=100, epsilon=0.01):
    """
    input:
    - policy: array[states X actions] with probabilities for each action
    - env: environment
    - gamma: discount_rate
    - n_iter: # of times to iteration
    output:
    - policy: optimal policy
    """
    # initialize state value function
    V = np.zeros(env.nS)
    for k in range(n_iter):
        v_delta = 0
        for s in range(env.nS):
            greedy_a, greedy_a_value = get_greedy_action(s, env, V, gamma)
            v_delta = max(v_delta, abs(greedy_a_value - V[s]))
            V[s] = greedy_a_value
            # can create policy from final V or just update it in this loop
            policy[s] = np.eye(env.nA)[greedy_a]

        if v_delta < epsilon:
            print(f'value iteration converged after {k+1} iterations')
            break
        elif k == n_iter - 1:
            print('value iteration never converged')

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

    optimal_policy, V = iterate_state_values(policy, env, gamma)

    print('optimal grid policy: 0=Left, 1=Down, 2=Right, 3=Up')
    print(optimal_policy.argmax(axis=1)
          .reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))))

    print('optimal state value function')
    print(V.reshape(int(np.sqrt(env.nS)), int(np.sqrt(env.nS))))

    env.close()


if __name__ == "__main__":
    main()
