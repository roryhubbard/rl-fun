import numpy as np


SEED = 0


def initialize_policy(n_actions, n_states, random=False):
    if random:
        policy = np.random.default_rng(SEED).dirichlet(
            np.ones(n_actions), size=n_states)
    else:
        # all actions are of equal probability
        policy = np.ones((n_states, n_actions)) / n_actions
    return policy


def get_deterministic_next_state(s, a, env):
    grid_dimension = int(np.sqrt(env.nS))
    next_state = 0
    if a == 0:
        if s % grid_dimension == 0:  # on left boundary
            next_state = s
        else:
            next_state = s - 1
    if a == 1:
        if s >= env.nS - grid_dimension:  # on bottom boundary
            next_state = s
        else:
            next_state = s + grid_dimension
    if a == 2:
        if (s - grid_dimension + 1) % grid_dimension == 0:  # on right boundary
            next_state = s
        else:
            next_state = s + 1
    if a == 3:
        if s < grid_dimension:  # on top boundary
            next_state = s
        else:
            next_state = s - grid_dimension
    return next_state


def get_state_action_value(s, env, V, a, gamma, make_deterministic=True):
    """
    input:
    - s: state
    - env: environment
    - V: current value function (expected reward at each state)
    - a: action
    - gamma: discount_rate
    output:
    - expected reward given state and action
    """
    state_action_value = 0
    for tran_p, next_state, r, done in env.P[s][a]:
        if make_deterministic:
            deterministic_next_state = get_deterministic_next_state(s, a, env)
            if next_state == deterministic_next_state:
                state_action_value = r + gamma * V[next_state]
                break
        else:
            state_action_value += tran_p * (r + gamma * V[next_state])
    return state_action_value


def get_greedy_action(s, env, V, gamma):
    """
    input:
    - s: state
    - env: environment
    - gamma: discount_rate
    output:
    - (action, action_value)
        - action that leads to largest expected reward
        - value associated to greedy action
    """
    action_space = np.zeros(env.nA)
    for a in range(env.nA):
        action_space[a] = get_state_action_value(s, env, V, a, gamma)
    return action_space.argmax(), action_space[action_space.argmax()]
