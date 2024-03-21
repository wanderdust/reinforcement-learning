import torch
import torch.nn as nn
from torch.nn import functional as F


class QFunction(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(QFunction, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_inputs, out_channels=16, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_actions)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
