import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import sys
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================
# REPRODUCIBILITY
# ============================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# CNN MODEL - Identical to base GitFL paper architecture
# ============================================================
class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
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
        x = self.fc3(x)
        return x


# ============================================================
# DATA DISTRIBUTION
# ============================================================
def iid_split(dataset, num_nodes, node_id, samples_per_node=500, seed=42):
    np.random.seed(seed)
    all_indices = np.random.permutation(len(dataset))
    start    = node_id * samples_per_node
    end      = start + samples_per_node
    selected = all_indices[start:end].tolist()
    return Subset(dataset, selected)


def non_iid_split(dataset, num_nodes, node_id, alpha=0.1, samples_per_node=500, seed=42):
    np.random.seed(seed)
    targets     = np.array(dataset.targets)
    num_classes = 10

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    node_indices  = [[] for _ in range(num_nodes)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions    = np.random.dirichlet([alpha] * num_nodes)
        counts         = (proportions * len(class_indices[c])).astype(int)
        counts[-1]     = len(class_indices[c]) - counts[:-1].sum()
        splits         = np.split(class_indices[c], np.cumsum(counts)[:-1])
        for n in range(num_nodes):
            node_indices[n].extend(splits[n].tolist())

    node_idx = node_indices[node_id]
    np.random.shuffle(node_idx)
    node_idx = node_idx[:samples_per_node]
    return Subset(dataset, node_idx)


# ============================================================
# NETWORK UTILITIES
# ============================================================
def send_data(sock, data_bytes):
    length = struct.pack('>I', len(data_bytes))
    sock.sendall(length + data_bytes)


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_data(sock):
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None
    total = struct.unpack('>I', raw_len)[0]
    return recv_exact(sock, total)


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred   = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================
# LOCAL TRAINING
# ============================================================
def train_local(model, train_loader, device, local_epochs, lr, momentum):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss   = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            epoch_loss    += loss.item()
            pred           = output.argmax(dim=1)
            epoch_correct += pred.eq(batch_target).sum().item()
            epoch_total   += batch_target.size(0)

        epoch_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        avg_loss  = epoch_loss / len(train_loader)
        print(f"        Epoch {epoch + 1}/{local_epochs}   Loss: {avg_loss:.4f}   Train Acc: {epoch_acc:.2f}%")

    return model


# ============================================================
# FEDAVG AGGREGATION
# ============================================================
def fedavg(model_a, model_b):
    state_a   = model_a.state_dict()
    state_b   = model_b.state_dict()
    avg_state = {}
    for key in state_a:
        avg_state[key] = (state_a[key].float() + state_b[key].float()) / 2.0
    result = copy.deepcopy(model_a)
    result.load_state_dict(avg_state)
    return result


# ============================================================
# WEIGHT EXCHANGE - Both nodes send and receive simultaneously
# ============================================================
def exchange_weights(node_id, my_ip, peer_ip, port, weights_bytes, wait_timeout=60):
    """
    Both nodes run a sender thread and a receiver server simultaneously.
    Each node listens on its own IP:port and connects to the peer's IP:port.
    Returns (received_bytes, exchange_duration_seconds).
    """
    import threading

    received_data = [None]
    recv_error    = [None]
    send_error    = [None]

    def receiver():
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((my_ip, port))
            srv.listen(1)
            srv.settimeout(wait_timeout)
            conn, addr = srv.accept()
            conn.settimeout(wait_timeout)
            raw = recv_data(conn)
            conn.close()
            srv.close()
            if raw:
                received_data[0] = raw
        except Exception as e:
            recv_error[0] = str(e)

    def sender():
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect((peer_ip, port))
                send_data(s, weights_bytes)
                s.close()
                return
            except Exception:
                time.sleep(2)
        send_error[0] = f"Could not connect to {peer_ip}:{port} within {wait_timeout}s"

    t_start     = time.time()
    recv_thread = threading.Thread(target=receiver, daemon=True)
    send_thread = threading.Thread(target=sender,   daemon=True)

    recv_thread.start()
    send_thread.start()

    recv_thread.join(timeout=wait_timeout + 5)
    send_thread.join(timeout=wait_timeout + 5)

    exch_time = time.time() - t_start

    if send_error[0]:
        raise RuntimeError(f"Send failed: {send_error[0]}")
    if recv_error[0]:
        raise RuntimeError(f"Receive failed: {recv_error[0]}")
    if received_data[0] is None:
        raise RuntimeError("No data received from peer.")

    return received_data[0], exch_time


# ============================================================
# PRINT HELPERS
# ============================================================
SEP  = "=" * 68
THIN = "-" * 68


def print_comparison_table(results, distribution, alpha, samples_per_node,
                            local_epochs, batch_size, num_rounds):

    print()
    print(SEP)
    print("  ROUND-WISE ACCURACY CHART")
    print(SEP)
    print()
    print(f"  {'Round':<8} {'Local Acc':>12} {'After FedAvg':>14} "
          f"{'Change':>10}  {'Comm(s)':>9}  {'Trend':>6}")
    print(f"  {'-'*66}")

    for r in results:
        sign  = "+" if r['delta'] >= 0 else ""
        arrow = "UP" if r['delta'] > 0.5 else ("DOWN" if r['delta'] < -0.5 else "FLAT")
        print(f"  {r['round']:<8} {r['local_acc']:>11.2f}%  {r['agg_acc']:>12.2f}%  "
              f"{sign}{r['delta']:>7.2f}%  "
              f"{r['comm_time']:>8.1f}s  {arrow:>6}")

    print()
    final_acc  = results[-1]['agg_acc']
    best_acc   = max(r['agg_acc'] for r in results)
    best_rnd   = max(results, key=lambda r: r['agg_acc'])['round']
    avg_comm   = sum(r['comm_time'] for r in results) / len(results)
    total_comm = sum(r['comm_time'] for r in results)

    print(f"  Final accuracy (Round {num_rounds})  : {final_acc:.2f}%")
    print(f"  Best accuracy (Round {best_rnd})    : {best_acc:.2f}%")
    print()
    print(f"  Avg comm time per round      : {avg_comm:.1f}s  (simultaneous P2P exchange)")
    print(f"  Total comm time (all rounds) : {total_comm:.1f}s")
    print()

    # ----------------------------------------------------------
    # COMPARISON WITH BASE PAPER
    # ----------------------------------------------------------
    print(SEP)
    print("  COMPARISON WITH BASE GitFL PAPER  (RTSS 2023)")
    print(SEP)
    print()

    paper_reference = {
        'iid': {
            'GitFL_resnet18': 72.40,
            'FedAvg_resnet18': 64.80,
            'note': 'ResNet-18, 100 clients, IID  (Table II, base paper)'
        },
        'non_iid_0.1': {
            'GitFL_resnet18': 54.85,
            'FedAvg_resnet18': 42.87,
            'note': 'ResNet-18, 100 clients, Non-IID alpha=0.1  (Table II, base paper)'
        },
        'non_iid_0.5': {
            'GitFL_resnet18': 70.36,
            'FedAvg_resnet18': 61.07,
            'note': 'ResNet-18, 100 clients, Non-IID alpha=0.5  (Table II, base paper)'
        }
    }

    if distribution == 'iid':
        ref_key = 'iid'
    elif alpha <= 0.15:
        ref_key = 'non_iid_0.1'
    else:
        ref_key = 'non_iid_0.5'

    ref = paper_reference[ref_key]

    print(f"  Scenario compared  :  {distribution.upper()}  (alpha={alpha})")
    print()
    print(f"  {'Metric':<40} {'Base Paper':>12} {'This Work':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Final Test Accuracy':<40} {'N/A (diff arch)':>12} {final_acc:>11.2f}%")
    print(f"  {'Best Test Accuracy (any round)':<40} {'N/A (diff arch)':>12} {best_acc:>11.2f}%")
    print(f"  {'GitFL accuracy (ResNet-18, ref only)':<40} {ref['GitFL_resnet18']:>11.2f}% {'--':>12}")
    print(f"  {'FedAvg accuracy (ResNet-18, ref only)':<40} {ref['FedAvg_resnet18']:>11.2f}% {'--':>12}")
    print()
    print(f"  Reference note  :  {ref['note']}")
    print()

    print(THIN)
    print("  SETUP COMPARISON")
    print(THIN)
    print()
    print(f"  {'Parameter':<35} {'Base GitFL Paper':>18} {'This Work':>14}")
    print(f"  {'-'*70}")
    print(f"  {'Dataset':<35} {'CIFAR-10':>18} {'CIFAR-10':>14}")
    print(f"  {'Model':<35} {'CNN / ResNet-18 / VGG':>18} {'CNN only':>14}")
    print(f"  {'Total clients / nodes':<35} {'100 clients':>18} {'2 nodes':>14}")
    print(f"  {'Active clients per round':<35} {'10 (10%)':>18} {'2 (100%)':>14}")
    print(f"  {'Samples per client':<35} {'500':>18} {str(samples_per_node):>14}")
    print(f"  {'Data distribution':<35} {'IID + Non-IID':>18} {distribution.upper():>14}")
    print(f"  {'Local epochs':<35} {'5':>18} {str(local_epochs):>14}")
    print(f"  {'Batch size':<35} {'50':>18} {str(batch_size):>14}")
    print(f"  {'Optimizer':<35} {'SGD lr=0.01 m=0.5':>18} {'SGD lr=0.01 m=0.5':>14}")
    print(f"  {'FL protocol':<35} {'Asynchronous':>18} {'Synchronous':>14}")
    print(f"  {'Hardware':<35} {'i9 + RTX 3090 GPU':>18} {'CPU only':>14}")
    print(f"  {'Deployment':<35} {'Simulation only':>18} {'2 physical VMs':>14}")
    print(f"  {'Uncertainty modeling':<35} {'Yes (Gaussian)':>18} {'No':>14}")
    print()
    print(SEP)


# ============================================================
# MAIN NODE LOGIC
# ============================================================
def run_node(node_id, my_ip, peer_ip,
             num_rounds=5,
             local_epochs=5,
             batch_size=50,
             samples_per_node=500,
             distribution='iid',
             alpha=0.1,
             port=8000,
             inter_round_wait=15):

    device = torch.device('cpu')

    # ----------------------------------------------------------
    # ONE-TIME HEADER
    # ----------------------------------------------------------
    print()
    print(SEP)
    print("  DECENTRALIZED FEDERATED LEARNING - NODE CONFIGURATION")
    print(SEP)
    print()
    print("  Node identity")
    print(f"      Node ID          : {node_id}")
    print(f"      My IP            : {my_ip}:{port}")
    print(f"      Peer IP          : {peer_ip}:{port}")
    print()
    print("  Experiment parameters  (aligned with base GitFL paper where possible)")
    print(f"      Rounds           : {num_rounds}")
    print(f"      Local epochs     : {local_epochs}   (base paper: 5)")
    print(f"      Batch size       : {batch_size}   (base paper: 50)")
    print(f"      Samples per node : {samples_per_node}   (base paper: 500 per client)")
    print(f"      Distribution     : {distribution.upper()}   alpha={alpha} for non-IID")
    print(f"      Optimizer        : SGD   lr=0.01   momentum=0.5   (base paper params)")
    print()
    print("  Model architecture  (identical to base paper)")
    print("      CNNCifar: Conv(3->6, 5x5) -> Pool -> Conv(6->16, 5x5) -> Pool")
    print("                -> FC(400->120) -> FC(120->84) -> FC(84->10)")
    print()
    print("  Deployment")
    print(f"      Device           : CPU (no GPU)")
    print(f"      Deployment type  : Two physical VMware VMs")
    print(f"      Inter-round wait : {inter_round_wait}s grace period between rounds")
    print()
    print("  Differences from base paper (acknowledged limitations)")
    print("      - 2 nodes vs 100 simulated clients")
    print("      - Synchronous protocol vs asynchronous GitFL")
    print("      - CPU only vs RTX 3090 GPU")
    print("      - No uncertainty/delay simulation")
    print()

    # ----------------------------------------------------------
    # DATASET
    # ----------------------------------------------------------
    print(THIN)
    print("  Loading and partitioning CIFAR-10 dataset...")
    print(THIN)
    print()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train = datasets.CIFAR10('./data/cifar10', train=True,  download=True, transform=transform)
    full_test  = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)

    if distribution == 'non_iid':
        train_subset = non_iid_split(full_train, 2, node_id, alpha=alpha,
                                     samples_per_node=samples_per_node)
        dist_label = f"Non-IID  Dirichlet alpha={alpha}"
    else:
        train_subset = iid_split(full_train, 2, node_id,
                                 samples_per_node=samples_per_node)
        dist_label = "IID  (random uniform split)"

    test_loader  = DataLoader(full_test,    batch_size=100,        shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    print(f"  Training samples for this node  : {len(train_subset)}")
    print(f"  Data distribution               : {dist_label}")
    print(f"  Test samples (full set)         : {len(full_test)}")
    print()

    model = CNNCifar(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total model parameters          : {total_params:,}")
    print()
    print("  Setup complete. Starting federated learning rounds.")
    print()

    # ----------------------------------------------------------
    # FL ROUNDS
    # ----------------------------------------------------------
    results = []

    for round_num in range(1, num_rounds + 1):

        print(SEP)
        print(f"  ROUND {round_num} / {num_rounds}")
        print(SEP)
        print()

        # --- Local Training ---
        print(f"  [1/3]  Local training  ({local_epochs} epochs, batch size {batch_size})")
        print()
        t_train_start = time.time()
        local_model   = copy.deepcopy(model)
        local_model   = train_local(local_model, train_loader, device,
                                    local_epochs, lr=0.01, momentum=0.5)
        train_time    = time.time() - t_train_start

        acc_local = evaluate(local_model, test_loader, device)
        print()
        print(f"        Training time          : {train_time:.1f}s")
        print(f"        Test accuracy (local)  : {acc_local:.2f}%")
        print()

        # --- Weight Exchange ---
        print(f"  [2/3]  Weight exchange with peer  (timeout: {inter_round_wait + 45}s)")
        print()
        weights_bytes = pickle.dumps(local_model.state_dict())
        print(f"        Serialized model size  : {len(weights_bytes):,} bytes")
        print(f"        Waiting for peer...    (both nodes send and receive simultaneously)")
        print()

        try:
            raw_peer, exch_time = exchange_weights(
                node_id, my_ip, peer_ip, port,
                weights_bytes,
                wait_timeout=inter_round_wait + 45
            )
            print(f"        Exchange complete      : {len(raw_peer):,} bytes received")
            print(f"        Exchange duration      : {exch_time:.1f}s")
        except RuntimeError as e:
            print(f"        ERROR during exchange  : {e}")
            print("        Stopping experiment.")
            break

        print()

        # --- Aggregation ---
        print(f"  [3/3]  FedAvg aggregation")
        print()
        peer_model = CNNCifar(num_classes=10).to(device)
        peer_model.load_state_dict(pickle.loads(raw_peer))
        model = fedavg(local_model, peer_model)

        acc_agg = evaluate(model, test_loader, device)
        delta   = acc_agg - acc_local
        sign    = "+" if delta >= 0 else ""

        print(f"        Test accuracy (aggregated) : {acc_agg:.2f}%")
        print(f"        Change vs local            : {sign}{delta:.2f}%")
        print()

        results.append({
            'round':     round_num,
            'local_acc': acc_local,
            'agg_acc':   acc_agg,
            'delta':     delta,
            'comm_time': exch_time   # simultaneous P2P exchange duration
        })

        # --- Inter-round wait ---
        if round_num < num_rounds:
            print(f"        Waiting {inter_round_wait}s before next round"
                  f"  (grace period for peer synchronization)...")
            time.sleep(inter_round_wait)
            print()

    # ----------------------------------------------------------
    # FINAL OUTPUT
    # ----------------------------------------------------------
    print_comparison_table(
        results, distribution, alpha,
        samples_per_node, local_epochs, batch_size, num_rounds
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':

    if len(sys.argv) < 4:
        print()
        print("  Usage:")
        print("    python3 fl_node.py <node_id> <my_ip> <peer_ip> [distribution] [alpha] [rounds]")
        print()
        print("  Arguments:")
        print("    node_id       : 0 or 1")
        print("    my_ip         : IP address of this machine")
        print("    peer_ip       : IP address of the peer machine")
        print("    distribution  : iid  or  non_iid   (default: iid)")
        print("    alpha         : Dirichlet alpha for non-IID  (default: 0.1)")
        print("    rounds        : number of FL rounds  (default: 5)")
        print()
        print("  Examples:")
        print("    python3 fl_node.py 0 192.168.221.128 192.168.221.129")
        print("    python3 fl_node.py 0 192.168.221.128 192.168.221.129 non_iid 0.1")
        print("    python3 fl_node.py 0 192.168.221.128 192.168.221.129 non_iid 0.5 10")
        print()
        sys.exit(1)

    node_id      = int(sys.argv[1])
    my_ip        = sys.argv[2]
    peer_ip      = sys.argv[3]
    distribution = sys.argv[4] if len(sys.argv) > 4 else 'iid'
    alpha        = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    num_rounds   = int(sys.argv[6])   if len(sys.argv) > 6 else 5

    run_node(
        node_id=node_id,
        my_ip=my_ip,
        peer_ip=peer_ip,
        num_rounds=num_rounds,
        local_epochs=5,
        batch_size=50,
        samples_per_node=500,
        distribution=distribution,
        alpha=alpha,
        port=8000,
        inter_round_wait=15
    )