import gym
import numpy as np
import torch
import torchvision.transforms as T
from utils import get_greedy_action, preprocess_frame, initialize_frame_sequence

import cv2
import time


Q = torch.load('DQN/trained_Q.pth')
Q.eval()
env = gym.make('Breakout-v0', frameskip=4)
env.reset()

m = 4
num_episodes = 2

transform = T.Compose([T.ToTensor()])

for _ in range(num_episodes):
    frame_sequence = initialize_frame_sequence(env, m)
    state = transform(np.stack(frame_sequence, axis=2))
    done = False

    while not done:
        action = get_greedy_action(
            Q, state.unsqueeze(0)).item()
        frame, reward, done, _ = env.step(action)

        frame_sequence.append(preprocess_frame(frame))
        state = transform(np.stack(frame_sequence, axis=2))

        env.render()
        time.sleep(.1)

        # cv2.imshow('', frame)
        # cv2.waitKey(100)
