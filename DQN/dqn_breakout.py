import random
from collections import deque
from copy import deepcopy
import gym
import numpy as np
import torch
import torch.optim as optim
from q_network import QNetwork
from utils import preprocess_frame, initialize_frame_sequence, \
    annealed_epsilon, get_epsilon_greedy_action, get_greedy_action


# ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


def deep_qlearning(env, nframes, discount_factor, N, C, mini_batch_size,
                   replay_start_size, sgd_update_frequency):
    """
    Input:
    - env: environment
    - nframes: # of frames to train on
    - discount_factor (gamma): how much to discount future rewards
    - N: replay memory size
    - C: number of steps before updating Q target network
    - mini_batch_size: mini batch size
    - replay_start_size: minimum size of replay memory before learning starts
    - sgd_update_frequency: number of action selections in between consecutive
      mini batch SGD updates

    Output:
    - Q: trained Q-network
    """
    initial_exploration = 1.
    final_exploration = 0.1
    # number of frames over which the epsilon is annealed to its final value
    final_exploration_frame = 1000000

    n_actions = env.action_space.n
    Q = QNetwork(n_actions).type(torch.FloatTensor)
    Q_target = deepcopy(Q)

    lr = 0.00025
    momentum = 0.95
    optimizer = optim.RMSprop(Q.parameters(), lr=lr, momentum=momentum)

    D = deque(maxlen=N)  # replay memory

    last_Q_target_update = 0

    m = 4  # number of consecutive frames to stack for input to Q network

    trained_frames_count = 0

    last_sgd_update = 0

    epsilon = .1
    while True:
        frame_sequence = initialize_frame_sequence(env, Q, m, epsilon)
        state = torch.as_tensor(np.stack(frame_sequence))
        done = False

        while not done:
            epsilon = annealed_epsilon(
                initial_exploration, final_exploration,
                final_exploration_frame, trained_frames_count)
            action = get_epsilon_greedy_action(Q, state, epsilon, n_actions)
            # action = env.action_space.sample()
            frame, reward, done, _ = env.step(action.item())

            reward = torch.tensor([reward])

            if done:
                next_state = None
            else:
                frame_sequence.append(preprocess_frame(frame))
                next_state = torch.as_tensor(np.stack(frame_sequence))

            # store transition in replay memory
            D.append((state, action, reward, next_state))

            state = next_state

            if len(D) < replay_start_size:
                continue

            last_sgd_update += 1
            if last_sgd_update < sgd_update_frequency:
                continue
            last_sgd_update = 0

            mini_batch = random.sample(D, mini_batch_size)
            mini_batch = list(zip(*mini_batch))

            non_final_mask = torch.tensor(
                tuple(map(lambda next_state: next_state is not None,
                          mini_batch[3])), dtype=torch.bool)
            non_final_next_states = torch.cat([
                next_state for next_state in mini_batch[3]
                if next_state is not None
            ])

            state_batch = torch.stack(mini_batch[0]).type(torch.FloatTensor)
            action_batch = torch.stack(mini_batch[1])
            reward_batch = torch.stack(mini_batch[2])

            state_action_values = Q(state_batch).gather(1, action_batch)

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
    discount_factor = 0.99
    N = 1000000  # replay memory size
    C = 10000  # number of steps before updating Q target network
    mini_batch_size = 32
    # minimum size of replay memory before learning starts
    # replay_start_size = 50000
    replay_start_size = 40
    # number of action selections in between consecutive mini batch SGD updates
    sgd_update_frequency = 4

    Q = deep_qlearning(env, nframes, discount_factor, N, C, mini_batch_size,
                       replay_start_size, sgd_update_frequency)

    # print(env.action_space.n)
    # print(env.observation_space.shape)
    # print(env.get_action_meanings())
    # print(env.get_keys_to_action())


if __name__ == "__main__":
    main()
