import random
from collections import deque
from copy import deepcopy
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


def deep_qlearning(env, nframes):
    """
    Input:
    - env: environment
    - nframes: # of frames to train on

    Output:
    - Q: trained Q-network
    """
    discount_factor = 0.99
    lr = 0.1

    initial_exploration = 1.
    final_exploration = 0.1

    Q_target = QNetwork()
    Q = QNetwork()

    N = 1000000  # replay memory size
    D = deque(maxlen=N)  # replay memory

    C = 10000  # number of iterations before resetting Q_target
    last_Q_target_update = 0
    m = 4  # number of consecutive frames to stack for input to Q network
    mini_batch_size = 32
    frame_sequence = deque(maxlen=m)

    trained_frames_count = 0

    sgd_update_frequency = 4
    replay_start_size = 50000

    last_sgd_update = 0

    while True:
        frame_sequence.append(preprocess_frame(env.reset()))  # 's' in paper
        frame_arr = None  # 'phi' in paper
        done = False

        while not done:
            # epsilon = some function of initial_exploration and final_exploration
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
            # store transition in replay memory
            D.append((frame_arr, action, reward, next_frame_arr, done))
            frame_arr = next_frame_arr

            if len(D) < replay_start_size:
                continue

            last_sgd_update += 1

            if last_sgd_update < sgd_update_frequency:
                continue

            last_sgd_update = 0

            rng = np.random.default_rng()
            mini_batch_idx = rng.choice(len(D), mini_batch_size)

            for idx in mini_batch_idx:
                transition = D[idx]
                frame_arr, action, reward, next_frame_arr, done = transition
                target = reward if done \
                    else reward + discount_factor \
                    * Q_target.get_greedy_action(next_frame_arr)

                loss = (target - Q.get_greedy_action(frame_arr))**2
                Q.gradient_descent_step(loss)

                last_Q_target_update += 1
                trained_frames_count += 1

                if last_Q_target_update % C == 0:
                    Q_target = deepcopy(Q)

                if trained_frames_count == nframes:
                    return


def main():
    # Note: setting frameskip to an int makes the game deterministic
    k = 4  # number of frames to skip before new action is selected
    env = gym.make('Breakout-v0', frameskip=4)

    nframes = 50000000  # train for a total of 50 million frames
    Q = deep_qlearning(env, nframes)

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
