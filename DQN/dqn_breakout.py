import random
import gym
from collections import deque
from q_network import QNetwork


SEED = 0


def deep_qlearning(env, Q, nepisodes):
    """
    Input:
    - env: environment
    - nepisodes: # of episodes to run

    Output:
    - Q: trained Q-network
    """
    discount_rate = 0.99
    lr = 0.1
    epsilon = 0.1

    Q_target = QNetwork()
    Q = QNetwork()

    N = 10  # replay buffer size
    D = deque(maxlen=N)  # replay buffer
    transitions = []

    C = 10  # number of iterations before resetting Q_target

    for episode in range(nepisodes):
        state = env.reset()
        done = False

        i = 0
        while not done:
            action = Q.get_epsilon_greedy_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            transitions.append((state, action, reward, next_state, done))

            mini_batch = []
            for transition in mini_batch:
                state, action, reward, next_state, done = transition
                y = reward if done \
                    else reward + discount_rate \
                    * Q_target.get_greedy_action(next_state)

                loss = (y - Q.get_greedy_action(state))**2
                Q.gradient_descent_step(loss)

            i += 1
            if i % C == 0:
                Q_target = Q.clone()


def main():
    env = gym.make('Breakout-v0')
    env.seed(SEED)
    env.reset()

    nepisodes = 100
    Q = deep_qlearning(env, nepisodes)

    # print(env.action_space.n)
    # print(env.observation_space.shape)
    # print(env.get_action_meanings())
    # print(env.get_keys_to_action())

    # for i in range(10):
    #     action = env.action_space.sample()
    #     print(action)
    #     state, reward, done, _ = env.step(action)
    #     env.render()



if __name__ == "__main__":
    main()
