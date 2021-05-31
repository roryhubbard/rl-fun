import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        # (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # compute 3136
        self.fl1 = nn.Linear(6 * 6 * 64, 512)
        self.fl2 = nn.Linear(512, n_actions)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.fl1(h.view(h.size(0), -1)))
        h = self.fl2(h)

        return h
