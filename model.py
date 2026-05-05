import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Lightweight CNN for CIFAR-10 (10 classes)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 16x16x32 -> 16x16x64
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16x16x32
        x = self.pool(F.relu(self.conv2(x)))  # -> 8x8x64
        x = x.view(-1, 64 * 8 * 8)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x