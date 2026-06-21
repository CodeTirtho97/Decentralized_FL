"""
shared/model.py  --  CNN model definition
                     CNNCifar: identical architecture to base GitFL paper (RTSS 2023)
                     Used by all experiments: centralized and decentralized.
"""

import torch.nn as nn
import torch.nn.functional as F


class CNNCifar(nn.Module):
    """
    Lightweight CNN for CIFAR-10.
    Architecture: Conv(3->6) -> Pool -> Conv(6->16) -> Pool -> FC(400->120->84->10)
    Total parameters: ~62,006
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
