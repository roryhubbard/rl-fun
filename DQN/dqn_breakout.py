import random
from collections import deque
import gym
import numpy as np
from q_network import QNetwork

import time
import cv2


"""
In these experiments, we used the RMSProp (see http://www.cs.toronto.edu/
,tijmen/csc321/slides/lecture_slides_lec6.pdf ) algorithm with minibatches of size
32. The behaviour policy during training was e-greedy with e annealed linearly
from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter. We trained
for a total of 50 million frames (that is, around 38 days of game experience in total)
and used a replay memory of 1 million most recent frames.
"""


def preprocess_frame(frame):
    """
    210 x 160 x 3 -> 80 x 80
    """
    grayscale = frame @ [0.2989, 0.5870, 0.1140]
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

    N = 1000000  # replay memory size
    D = deque(maxlen=N)  # replay memory

    C = 10  # number of iterations before resetting Q_target
    m = 4  # number of consecutive frames to stack for input to Q network
    frame_sequence = deque(maxlen=m)

    for episode in range(nepisodes):
        frame_sequence.append(preprocess_frame(env.reset()))  # s in paper
        frame_arr = None  # phi in paper
        done = False

        i = 0
        while not done:
            # action = Q.get_epsilon_greedy_action(state, epsilon)
            action = env.action_space.sample()
            frame, reward, done, _ = env.step(action)
            frame_sequence.append(preprocess_frame(frame))

            if len(frame_sequence) < m:
                continue

            if frame_arr is None:
                frame_arr = np.stack(frame_sequence)
                continue

            next_frame_arr = np.stack(frame_sequence)
            D.append((frame_arr, action, reward, next_frame_arr, done))
            frame_arr = next_frame_arr

            if len(D) < N:
                continue

            mini_batch = []  # size 32
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
            # cv2.imshow('', preprocess_frame(next_state))
            # cv2.waitKey(1)
            # time.sleep(.05)

        break


def main():
    # Note: setting frameskip to an int makes the game deterministic
    k = 4  # number of frames to skip before new action is selected
    env = gym.make('Breakout-v0', frameskip=4)

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
