from collections import deque
import random
import pickle
import torch


def preprocess_frame(frame):
    """
    210 x 160 x 3 -> 80 x 80
    Paper uses 84 x 84 but this is more convenient
    """
    grayscale = frame @ [0.2989, 0.5870, 0.1140]
    cropped = grayscale[35:195]
    downsampled = cropped[::2,::2]
    return downsampled


def initialize_frame_sequence(env, m):
    """
    Input:
    - env: environment
    - m: number of consecutive frames to stack for input to Q network

    Output:
    - frame_sequence: sequence of preprocessed observed frames from actions
    """
    frame_sequence = deque(maxlen=m)
    frame_sequence.append(preprocess_frame(env.reset()))

    for _ in range(m-1):
        action = env.action_space.sample()
        frame, _, done, _ = env.step(action)

        # if episode finishes before sequence is filled, try again
        if done:
            return initialize_frame_sequence(env, m)

        frame_sequence.append(preprocess_frame(frame))

    return frame_sequence


def annealed_epsilon(initial_exploration, final_exploration,
                     final_exploration_frame, frames_count):
    """
    Return linearly annealed exploration value
    """
    return initial_exploration - (initial_exploration - final_exploration) \
        * frames_count / final_exploration_frame


def get_greedy_action(Q, state):
    with torch.no_grad():
        return Q(state).max(1)[1].view(1)


def get_epsilon_greedy_action(Q, state, epsilon, n_actions):
    """
    Select a random action with probabily of epsilon or select the greedy
    action according to Q

    Input:
    - Q: network that maps states to actions
    - state: current state as given by the environment
    - epsilon: probability of exploring
    - n_actions: action space cardinality

    Output:
    - action
    """
    return torch.tensor([random.choice(tuple(range(n_actions)))],
                        dtype=torch.int64) \
        if random.uniform(0.0, 1.0) < epsilon \
        else get_greedy_action(Q, state)


def save_stuff(Q, episode_lengths):
    torch.save(Q, 'DQN/trained_Q.pth')
    with open("DQN/episode_lengths.txt", "wb") as f:
        pickle.dump(episode_lengths, f)
