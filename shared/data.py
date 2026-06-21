"""
shared/data.py  --  CIFAR-10 loading and data distribution
                    IID and Non-IID (Dirichlet) splits for 8-node experiments.
                    Same splits used by both centralized and decentralized code.
"""

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

NUM_NODES = 8

# Fix all random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_cifar10(root='./data'):
    """Load the pre-downloaded CIFAR-10 train/test sets with standard normalisation (download=False)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10(root, train=True,  download=False, transform=transform)
    test  = datasets.CIFAR10(root, train=False, download=False, transform=transform)
    return train, test


def iid_split(dataset, node_id, samples_per_node=2500, seed=42):
    """IID partition: shuffle all indices with a fixed seed, then hand each node a disjoint contiguous slice."""
    np.random.seed(seed)
    all_idx = np.random.permutation(len(dataset))
    start   = node_id * samples_per_node
    end     = start + samples_per_node
    return Subset(dataset, all_idx[start:end].tolist())


def non_iid_split(dataset, node_id, alpha=0.5, samples_per_node=2500, seed=42):
    """Non-IID partition: split each class across nodes by a Dirichlet(alpha) draw (lower alpha = more skew)."""
    np.random.seed(seed)
    targets   = np.array(dataset.targets)
    class_idx = [np.where(targets == c)[0] for c in range(10)]
    node_idx  = [[] for _ in range(NUM_NODES)]

    for c in range(10):
        np.random.shuffle(class_idx[c])
        props      = np.random.dirichlet([alpha] * NUM_NODES)
        counts     = (props * len(class_idx[c])).astype(int)
        counts[-1] = len(class_idx[c]) - counts[:-1].sum()
        splits     = np.split(class_idx[c], np.cumsum(counts)[:-1])
        for n in range(NUM_NODES):
            node_idx[n].extend(splits[n].tolist())

    chosen = node_idx[node_id]
    np.random.shuffle(chosen)
    return Subset(dataset, chosen[:samples_per_node])


def get_loaders(node_id, distribution, alpha, samples_per_node, batch_size):
    """
    Returns (train_loader, test_loader, num_train_samples, dist_label).
    Single call used by all client/node scripts.
    """
    train_full, test_full = load_cifar10()

    if distribution == 'non_iid':
        subset = non_iid_split(train_full, node_id, alpha, samples_per_node)
        label  = f"Non-IID  Dirichlet alpha={alpha}"
    else:
        subset = iid_split(train_full, node_id, samples_per_node)
        label  = "IID  (random uniform)"

    train_loader = DataLoader(subset,    batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_full, batch_size=100,        shuffle=False, num_workers=0)
    return train_loader, test_loader, len(subset), label
