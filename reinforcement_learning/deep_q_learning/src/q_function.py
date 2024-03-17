from torch import nn


def q_function(n_inputs, n_actions):
    return nn.Sequential(
        nn.Conv2d(in_channels=n_inputs, out_channels=16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=32 * 9 * 9, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=n_actions),
    )
