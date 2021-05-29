import random
import gym
from collections import deque
from q_network import QNetwork

import time
import cv2
import numpy as np


SEED = 0


def preprocess_image(img):
    """
    210 x 160 x 3 -> 80 x 80
    """
    grayscale = img @ [0.2989, 0.5870, 0.1140]
    cropped = grayscale[35:195]
    downsampled = cropped[::2,::2]
    return downsampled


def deep_qlearning(env, nepisodes):
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

    C = 10  # number of iterations before resetting Q_target
    m = 4  # number of consecutive images to stack for input to Q network
    k = 4  # number of frames to skip before new action is selected

    for episode in range(nepisodes):
        state = env.reset()
        done = False

        i = 0
        while not done:
            # action = Q.get_epsilon_greedy_action(state, epsilon)
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            D.append((state, action, reward, next_state, done))

            mini_batch = []
            for transition in mini_batch:
                state, action, reward, next_state, done = transition
                y = reward if done \
                    else reward + discount_rate \
                    * Q_target.get_greedy_action(next_state)

                loss = (y - Q.get_greedy_action(state))**2
                Q.gradient_descent_step(loss)

            # i += 1
            # if i % C == 0:
            #     Q_target = Q.clone()


            # env.render()
            # cv2.imshow('', preprocess_image(next_state))
            # cv2.waitKey(1)
            # time.sleep(.05)

        break


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
