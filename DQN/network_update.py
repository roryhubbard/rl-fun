import random
import torch
import torch.nn as nn


def sgd_update(Q, Q_target, D, mini_batch_size, discount_factor, optimizer):
    mini_batch = random.sample(D, mini_batch_size)
    mini_batch = list(zip(*mini_batch))

    non_final_mask = torch.tensor(
        tuple(map(lambda next_state: next_state is not None,
                  mini_batch[3])), dtype=torch.bool)
    non_final_next_states = torch.stack([
        next_state for next_state in mini_batch[3]
        if next_state is not None
    ]).type(torch.FloatTensor)

    state_batch = torch.stack(mini_batch[0])
    action_batch = torch.stack(mini_batch[1])
    reward_batch = torch.cat(mini_batch[2])

    state_action_values = Q(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(mini_batch_size)
    next_state_values[non_final_mask] = Q_target(
        non_final_next_states).max(1)[0].detach()

    y = reward_batch + discount_factor * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, y.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in Q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
