"""
fedstellar_experiment/shared/model_lightning.py

CNNCifar wrapped as a PyTorch Lightning module for use with p2pfl.
Identical architecture to shared/model.py — same conv layers, FC layers, ~62K params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class CNNCifarLightning(LightningModule):
    """
    CNNCifar for CIFAR-10.
    Architecture: Conv(3->6) -> Pool -> Conv(6->16) -> Pool -> FC(400->120->84->10)
    Parameters: ~62,006  (identical to shared/model.py CNNCifar)
    lr=0.01, momentum=0.5 — same as node.py train_local().
    """

    def __init__(self, num_classes=10, lr=0.01, momentum=0.5):
        super().__init__()
        self.save_hyperparameters()
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
        )
