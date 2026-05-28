"""
experiments/fedstellar/shared/data_fedstellar.py

CIFAR-10 IID partition for p2pfl.
Uses the same IID split logic as src/shared/data.py so results are directly
comparable to Experiment 3 (synchronous ring gossip).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SAMPLES_PER_NODE = 6250
DATA_ROOT        = './data'


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_fedstellar_partition(node_id: int, num_nodes: int = 8,
                              samples_per_node: int = SAMPLES_PER_NODE,
                              data_root: str = DATA_ROOT):
    """
    Returns (train_subset, test_dataset) using the same IID split as
    src/shared/data.py iid_split() -- same seed, same index order.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    transform  = _transform()
    train_full = datasets.CIFAR10(data_root, train=True,  download=False, transform=transform)
    test_full  = datasets.CIFAR10(data_root, train=False, download=False, transform=transform)

    all_idx      = np.random.permutation(len(train_full))
    start        = node_id * samples_per_node
    end          = start   + samples_per_node
    train_subset = Subset(train_full, all_idx[start:end].tolist())

    return train_subset, test_full


def make_test_loader(test_dataset, batch_size=100):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
