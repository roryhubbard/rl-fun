import gym
import numpy as np
import torch
from utils import get_greedy_action, preprocess_frame, initialize_frame_sequence

import cv2
import time


Q = torch.load('DQN/trained_Q.pth')
Q.eval()
env = gym.make('Breakout-v0')

m = 4
num_episodes = 2

for _ in range(num_episodes):
    frame_sequence = initialize_frame_sequence(env, m)
    state = torch.as_tensor(
        np.stack(frame_sequence)).type(torch.FloatTensor)
    done = False

    while not done:
        action = get_greedy_action(
            Q, state.unsqueeze(0))
        frame, reward, done, _ = env.step(action.item())

        frame_sequence.append(preprocess_frame(frame))
        state = torch.as_tensor(
            np.stack(frame_sequence)).type(torch.FloatTensor)

        env.render()
        time.sleep(.1)
        # cv2.imshow('', frame)
        # cv2.waitKey(100)
