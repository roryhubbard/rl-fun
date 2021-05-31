from collections import deque
import numpy as np
import gym
import torch
import torch.optim as optim
from q_network import QNetwork
from utils import preprocess_frame, initialize_frame_sequence, \
    annealed_epsilon, get_epsilon_greedy_action, save_stuff
from network_update import sgd_update


# ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


def deep_qlearning(env, nframes, discount_factor, N, C, mini_batch_size,
                   replay_start_size, sgd_update_frequency, initial_exploration,
                   final_exploration, final_exploration_frame, lr, momentum, m):
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
    - initial_exploration: initial epsilon value
    - final_exploration: final epsilon value
    - final_exploration_frame: number of frames over which the epsilon is
      annealed to its final value
    - lr: learning rate used by RMSprop
    - momentum: momentum value used by RMSprop
    - m: number of consecutive frames to stack for input to Q network

    Output:
    - Q: trained Q-network
    """
    n_actions = env.action_space.n
    Q = QNetwork(n_actions)
    Q_target = QNetwork(n_actions)
    Q_target.load_state_dict(Q.state_dict())
    Q_target.eval()

    optimizer = optim.RMSprop(Q.parameters(), lr=lr, momentum=momentum)

    D = deque(maxlen=N)  # replay memory

    last_Q_target_update = 0
    frames_count = 0
    last_sgd_update = 0
    episodes_count = 0
    episode_lengths = []

    while True:
        frame_sequence = initialize_frame_sequence(env, m)
        state = torch.as_tensor(
            np.stack(frame_sequence)).type(torch.FloatTensor)

        episode_length = 0
        done = False

        while not done:
            epsilon = annealed_epsilon(
                initial_exploration, final_exploration,
                final_exploration_frame, frames_count)
            action = get_epsilon_greedy_action(
                Q, state.unsqueeze(0), epsilon, n_actions)
            frame, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])

            episode_length += 1
            if done:
                next_state = None
                episode_lengths.append(episode_length)
            else:
                frame_sequence.append(preprocess_frame(frame))
                next_state = torch.as_tensor(
                    np.stack(frame_sequence)).type(torch.FloatTensor)

            # store transition in replay memory
            D.append((state, action, reward, next_state))

            state = next_state

            if len(D) < replay_start_size:
                continue

            last_sgd_update += 1
            if last_sgd_update < sgd_update_frequency:
                continue
            last_sgd_update = 0

            sgd_update(Q, Q_target, D, mini_batch_size,
                       discount_factor, optimizer)

            last_Q_target_update += 1
            frames_count += mini_batch_size

            if last_Q_target_update % C == 0:
                Q_target.load_state_dict(Q.state_dict())

            if frames_count >= nframes:
                return Q, episode_lengths

        episodes_count += 1
        if episodes_count % 100 == 0:
            save_stuff(Q, episode_lengths)
            print(f'episodes completed = {episodes_count},',
                  f'frames processed = {frames_count}')


def main():
    # Note: setting frameskip to an int makes the game deterministic
    k = 4  # number of frames to skip before new action is selected
    env = gym.make('Breakout-v0', frameskip=4)
    # print(env.get_action_meanings())
    # print(env.get_keys_to_action())

    nframes = 50000000  # train for a total of 50 million frames
    discount_factor = 0.99
    N = 1000000  # replay memory size
    C = 10000  # number of steps before updating Q target network
    mini_batch_size = 32
    replay_start_size = 50000  # minimum size of replay memory before learning starts
    sgd_update_frequency = 4  # number of action selections in between consecutive SGD updates
    initial_exploration = 1.  # initial epsilon valu3
    final_exploration = 0.1  # final epsilon value
    final_exploration_frame = 1000000  # number of frames over which the epsilon is annealed to its final value
    lr = 0.00025  # learning rate used by RMSprop
    momentum = 0.95  # momentum value used by RMSprop
    m = 4  # number of consecutive frames to stack for input to Q network

    Q, episode_lengths = deep_qlearning(
        env, nframes, discount_factor, N, C, mini_batch_size, replay_start_size,
        sgd_update_frequency, initial_exploration, final_exploration,
        final_exploration_frame, lr, momentum, m)

    save_stuff(Q, episode_lengths)


if __name__ == "__main__":
    main()
