import random
import numpy as np


def get_greedy_action(Q, observation):
    return Q.evaluate_state(observation).argmax()


def get_epsilon_greedy_action(env, Q, observation, epsilon):
    """
    Select a random action with probabily of epsilon or select the greedy
    action according to Q

    Input:
    - env: environment
    - Q: state-action value function
    - observation: current observation as given by the environment
    - epsilon: probability of "exploring" or choosing a random action

    Output:
    - action
    """
    return random.choice(tuple(range(env.action_space.n))) \
        if random.uniform(0.0, 1.0) < epsilon \
        else get_greedy_action(Q, observation)


def watch_greedy_policy(env, Q, max_num_steps=50):
    """
    Render environment while taking greedy actions according to Q
    """
    state = env.reset()
    env.render()
    done = False

    step_counter = 0
    while not done:
        action = get_greedy_action(Q, state)
        state, _, done, _ = env.step(action)
        env.render()

        step_counter += 1
        if step_counter > max_num_steps:
            print(f'failed to reach goal within {max_num_steps} steps')
            break
