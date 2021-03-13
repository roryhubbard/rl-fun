import gym
import numpy as np
from utils import initialize_policy, get_state_action_value


SEED = 0
# 0=Left, 1=Down, 2=Right, 3=Up


def get_updated_state_value(s, policy, env, V, gamma):
    """
    input:
    - s: state
    - policy: array[states X actions] with probabilities for each action
    - env: environment
    - V: current value function (expected reward at each state)
    - gamma: discount_rate
    output:
    - v: state value
    """
    v = 0
    for a, a_p in enumerate(policy[s]):
        v += a_p * get_state_action_value(s, env, V, a, gamma)
    return v


def evaluate_policy(policy, env, gamma, n_evals=100, epsilon=0.01):
    """
    input:
    - policy: array[states X actions] with probabilities for each action
    - env: environment
    - gamma: discount_rate
    - n_evals: # of times to evaluate policy
    - epsilon: convergence check threshold for state values
    output:
    - V: state value function 
    """
    # initialize state value function
    V = np.zeros(env.nS)
    for k in range(n_evals):
        v_delta = 0
        for s in range(env.nS):
            v_updated = get_updated_state_value(s, policy, env, V, gamma)
            v_delta = max(v_delta, abs(v_updated - V[s]))
            V[s] = v_updated

        if v_delta < epsilon:
            print(f'policy evaluation converged after {k+1} iterations')
            break
        elif k == n_evals - 1:
            print('policy evaluation never converged')

    return V


def main():
    env = gym.make('FrozenLake-v0')
    env.seed(SEED)
    env.reset()
    env.render()

    policy = initialize_policy(env.nA, env.nS)

    # set discount-rate [0, 1)
    gamma = 0.9

    v = evaluate_policy(policy, env, gamma)

    print('State Value Function After Policy Evaluation')
    print(v.reshape(env.nrow, env.ncol))

    env.close()


if __name__ == "__main__":
    main()
