"""
shared/train.py  --  Training, evaluation, and aggregation
                     All model update logic lives here.
"""

import copy
import torch
import torch.nn as nn
from shared.log import log


def train_local(model, train_loader, device, local_epochs, lr=0.01, momentum=0.5):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        e_loss = e_correct = e_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            e_loss    += loss.item()
            e_correct += out.argmax(dim=1).eq(target).sum().item()
            e_total   += target.size(0)

        e_acc = 100.0 * e_correct / e_total if e_total > 0 else 0.0
        log(f"        Epoch {epoch+1}/{local_epochs}"
            f"  |  Loss: {e_loss / len(train_loader):.4f}"
            f"  |  Train Acc: {e_acc:.2f}%")

    return model


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred     = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def fedavg(state_dicts):
    """Average N model state dicts. Used by centralized server and gossip blend."""
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = avg[key].float()
        for i in range(1, len(state_dicts)):
            avg[key] += state_dicts[i][key].float()
        avg[key] /= len(state_dicts)
    return avg


def blend_models(model_a, model_b, alpha=0.5):
    """Weighted average of two models."""
    sa      = model_a.state_dict()
    sb      = model_b.state_dict()
    blended = {k: alpha * sa[k].float() + (1.0 - alpha) * sb[k].float() for k in sa}
    result  = copy.deepcopy(model_a)
    result.load_state_dict(blended)
    return result
