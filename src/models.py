import torch
import torch.nn as nn


class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x