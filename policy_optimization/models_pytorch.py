import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Actor(nn.Module):

  def __init__(self, nstates, nactions):
    super().__init__()
    self.h1 = nn.Linear(nstates, 64)
    self.h2 = nn.Linear(64, 64)
    self.h3 = nn.Linear(64, nactions)

    log_std = -0.5 * np.ones(nactions, dtype=np.float32)
    self.log_std = nn.Parameter(torch.as_tensor(log_std))

  def forward(self, x, action=None):
    h = torch.tanh(self.h1(x))
    h = torch.tanh(self.h2(h))
    mu = self.h3(h).squeeze()
    std = torch.exp(self.log_std)
    pd = Normal(mu, std)
    if action is None:
      action = pd.sample()
    return action, pd.log_prob(action)


class Critic(nn.Module):

  def __init__(self, nstates):
    super().__init__()
    self.h1 = nn.Linear(nstates, 64)
    self.h2 = nn.Linear(64, 64)
    self.h3 = nn.Linear(64, 1)

  def forward(self, x):
    h = torch.relu(self.h1(x))
    h = torch.relu(self.h2(h))
    return self.h3(h).squeeze()

