# Federated Learning: Centralized vs. Decentralized — 8-Node AWS Experiment

A thesis implementation comparing centralized and decentralized federated learning architectures on CIFAR-10, deployed across 8 AWS EC2 instances in a ring topology.

---

## Overview

This project implements and compares two federated learning (FL) strategies:

- **Centralized FL** — A single aggregation server collects model weights from all clients, runs FedAvg, and broadcasts the result. Simple and accurate, but has a Single Point of Failure (SPOF).
- **Decentralized Async FL** — No server. Nodes train locally and push weights to their two ring neighbors. Blending happens asynchronously in a background thread. Resilient to node failures.

The comparison covers accuracy, communication cost, convergence speed, and fault tolerance under both IID and Non-IID data distributions.

---

## What This Extends (Base Paper)

This work builds on the **GitFL** architecture (gossip-based, asynchronous FL in a ring topology), adapting it for:

| Aspect | Base GitFL Paper | This Work |
|---|---|---|
| Nodes | Variable | Fixed 8-node ring on AWS |
| Infrastructure | Simulated | Real distributed cloud (AWS EC2) |
| Comparison baseline | None / minimal | Full centralized FL baseline |
| Data distribution | IID only | IID and Non-IID (Dirichlet α=0.5) |
| Fault tolerance | Mentioned | Automated demo (Exp 5-A and 5-B) |
| Code structure | Research prototype | SOLID-structured, clean entry points |

---

## Architecture

### Centralized FL (Experiments 1, 2, 5-B)

```
          Node 0  (SERVER)
         /    |    \
       Node1 Node2 ... Node7  (CLIENTS)
```

- Node 0 listens on port 9000
- All 7 clients upload → server aggregates (FedAvg) → broadcasts back
- If Node 0 goes offline, all 7 clients stall permanently

### Decentralized Async FL (Experiments 3, 4, 5-A)

```
          Node 0
         /       \
     Node 7      Node 1
       |               |
     Node 6      Node 2
         \       /
    Node 5 -- Node 4 -- Node 3
```

- Ring topology: each node communicates only with its left and right neighbors
- Each node listens on port `8000 + node_id`
- Background listener thread blends incoming weights asynchronously
- If one node fails, its neighbors detect the push failure, log it, and continue

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | PyTorch (CPU-only) |
| Dataset | CIFAR-10 (via torchvision) |
| Model | CNNCifar (~62K params) |
| Networking | Python `socket` (raw TCP) |
| Concurrency | `threading` (listener + push threads) |
| Cloud | AWS EC2 t3.micro, Ubuntu 24.04 LTS |
| Connectivity | AWS VPC private IPs (no charges) |
| Session mgmt | tmux |

---

## Model Architecture

```
CNNCifar (identical to base GitFL paper)
  Conv2d(3, 6, 5)  → ReLU → MaxPool(2,2)
  Conv2d(6, 16, 5) → ReLU → MaxPool(2,2)
  Linear(400, 120) → ReLU
  Linear(120, 84)  → ReLU
  Linear(84, 10)
  Total: ~62,006 parameters
```

---

## Project Structure

```
VM_Decentralized/
├── server.py           # Centralized server  (Experiments 1, 2, 5-B)
├── client.py           # Centralized client  (Experiments 1, 2, 5-B)
├── node.py             # Decentralized node  (Experiments 3, 4, 5-A)
├── shared/
│   ├── __init__.py
│   ├── log.py          # Logging utilities (SEP, THIN, log, log_thin)
│   ├── model.py        # CNNCifar definition
│   ├── data.py         # CIFAR-10 loading, IID / Non-IID splits
│   ├── net.py          # send_data, recv_data, make_server_socket
│   └── train.py        # train_local, evaluate, fedavg, blend_models
└── fl_node.py          # Original 2-node VM baseline (kept for reference)
```

---

## Experiments

| # | Architecture | Distribution | Notes |
|---|---|---|---|
| 1 | Centralized | IID | Baseline accuracy benchmark |
| 2 | Centralized | Non-IID (α=0.5) | Effect of data heterogeneity |
| 3 | Decentralized Async | IID | Gossip ring, no server |
| 4 | Decentralized Async | Non-IID (α=0.5) | Gossip ring, heterogeneous data |
| 5-A | Decentralized | Non-IID | Node 3 auto-exits at round 16; others finish |
| 5-B | Centralized | Non-IID | Server auto-exits at round 16; all clients halt |

---

## Default Hyperparameters

| Parameter | Value |
|---|---|
| FL rounds | 30 |
| Local epochs per round | 5 |
| Batch size | 64 |
| Samples per node | 2,500 |
| Optimizer | SGD (lr=0.01, momentum=0.5) |
| Gossip blend factor (α) | 0.5 |
| Dirichlet alpha (non-IID) | 0.5 |
| Fault demo trigger round | 16 |

---

## AWS Replication Guide

Follow these steps to replicate the full 8-node setup on your own AWS account.

### Step 1 — Create EC2 Instances

1. Go to AWS Console → EC2 → Launch Instances
2. Launch **8 instances** with these settings:
   - AMI: **Ubuntu Server 24.04 LTS** (Free Tier eligible)
   - Instance type: **t3.micro** (Free Tier eligible)
   - Number of instances: 8
   - Key pair: Create new → RSA → .pem → download and save safely
   - Network: Same VPC, same subnet (e.g., us-east-1a)
   - Auto-assign public IP: Enabled
3. Name the instances `fl-node-0` through `fl-node-7`

### Step 2 — Create Security Group

1. EC2 → Security Groups → Create Security Group
   - Name: `fl-nodes-sg`
   - Add inbound rule: SSH, port 22, Source: `0.0.0.0/0`
   - Create the group (do NOT add the self-reference yet)
2. After creation, edit inbound rules again:
   - Add rule: All TCP, port range `0-65535`, Source: **the security group itself** (type the SG name or ID in the source box)
   - This allows all 8 nodes to communicate freely with each other
3. Assign this security group to all 8 instances

### Step 3 — Record Node IPs

For each instance, note the **private IP** (used for FL communication) and **public IP** (used only for SSH from your laptop).

Private IPs stay fixed. Public IPs change on restart — re-check them when needed.

### Step 4 — Fix .pem Key Permissions (Windows)

Run **PowerShell as Administrator**:

```powershell
$pem = "C:\path\to\your-key.pem"
icacls $pem /inheritance:r
icacls $pem /remove "NT AUTHORITY\Authenticated Users"
icacls $pem /remove "BUILTIN\Users"
icacls $pem /grant:r "${env:USERNAME}:R"
```

Verify with `icacls $pem` — only your username should appear.

### Step 5 — SSH Into Each Node

```bash
ssh -i "C:\path\to\your-key.pem" ubuntu@<PUBLIC_IP>
```

Repeat for all 8 nodes. Keep separate terminal windows or use tmux.

### Step 6 — Install Python Environment (run on ALL 8 nodes)

```bash
sudo apt update -y && sudo apt install python3-pip python3-venv tmux -y
mkdir -p ~/fl_project/data
cd ~/fl_project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip cache purge
pip install torch torchvision numpy --index-url https://download.pytorch.org/whl/cpu
```

> **Important**: Always use `--index-url https://download.pytorch.org/whl/cpu`.
> The default pip install downloads the CUDA build (~2.5 GB) which will fill the 8 GB disk on t3.micro.
> The CPU-only build is ~200 MB and correct for t3.micro (no GPU).

Verify:
```bash
python3 -c "import torch; import torchvision; print('OK'); print(torch.__version__)"
```

### Step 7 — Transfer Code Files (run from your laptop)

```bash
KEY="C:\path\to\your-key.pem"
for IP in <NODE0_PUBLIC> <NODE1_PUBLIC> ... <NODE7_PUBLIC>; do
  scp -i "$KEY" server.py client.py node.py ubuntu@$IP:~/fl_project/
  scp -i "$KEY" -r shared/ ubuntu@$IP:~/fl_project/
done
```

Or transfer to each node individually with `scp`.

### Step 8 — Pre-download CIFAR-10 (run on ALL 8 nodes)

```bash
cd ~/fl_project && source venv/bin/activate
python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('CIFAR-10 ready')
"
```

### Step 9 — Use tmux for Long Sessions

```bash
tmux new -s fl        # start new session named 'fl'
# run experiment inside tmux
# Ctrl+B then D       → detach (experiment keeps running)
tmux attach -t fl     # reattach later
```

---

## Running the Experiments

Substitute your actual private IPs wherever shown. All commands run inside the venv:

```bash
cd ~/fl_project && source venv/bin/activate
```

---

### Experiment 1 — Centralized | IID

**Node 0 (server):**
```bash
python3 server.py <NODE0_PRIVATE_IP>
```

**Nodes 1-7 (clients) — run simultaneously:**
```bash
python3 client.py 1 <NODE0_PRIVATE_IP>
python3 client.py 2 <NODE0_PRIVATE_IP>
python3 client.py 3 <NODE0_PRIVATE_IP>
python3 client.py 4 <NODE0_PRIVATE_IP>
python3 client.py 5 <NODE0_PRIVATE_IP>
python3 client.py 6 <NODE0_PRIVATE_IP>
python3 client.py 7 <NODE0_PRIVATE_IP>
```

---

### Experiment 2 — Centralized | Non-IID

**Node 0 (server):**
```bash
python3 server.py <NODE0_PRIVATE_IP>
```

**Nodes 1-7 (clients):**
```bash
python3 client.py 1 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 2 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 3 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 4 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 5 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 6 <NODE0_PRIVATE_IP> --dist non_iid
python3 client.py 7 <NODE0_PRIVATE_IP> --dist non_iid
```

---

### Experiment 3 — Decentralized Async | IID

All 8 nodes run simultaneously. The ring order is:
`Node 0 ↔ Node 1 ↔ Node 2 ↔ ... ↔ Node 7 ↔ Node 0`

Command format: `python3 node.py <id> <my_ip> <left_ip> <right_ip>`

```bash
# Run each command on its respective node
python3 node.py 0 <NODE0_IP> <NODE7_IP> <NODE1_IP>
python3 node.py 1 <NODE1_IP> <NODE0_IP> <NODE2_IP>
python3 node.py 2 <NODE2_IP> <NODE1_IP> <NODE3_IP>
python3 node.py 3 <NODE3_IP> <NODE2_IP> <NODE4_IP>
python3 node.py 4 <NODE4_IP> <NODE3_IP> <NODE5_IP>
python3 node.py 5 <NODE5_IP> <NODE4_IP> <NODE6_IP>
python3 node.py 6 <NODE6_IP> <NODE5_IP> <NODE7_IP>
python3 node.py 7 <NODE7_IP> <NODE6_IP> <NODE0_IP>
```

Each node listens on port `8000 + node_id`. Node 0 waits for 0 seconds before first push, Node 1 waits 3s, Node 2 waits 6s, etc. This stagger ensures all listeners are up before any node pushes.

---

### Experiment 4 — Decentralized Async | Non-IID

Same as Experiment 3, but add `--dist non_iid` to all commands:

```bash
python3 node.py 0 <NODE0_IP> <NODE7_IP> <NODE1_IP> --dist non_iid
python3 node.py 1 <NODE1_IP> <NODE0_IP> <NODE2_IP> --dist non_iid
python3 node.py 2 <NODE2_IP> <NODE1_IP> <NODE3_IP> --dist non_iid
python3 node.py 3 <NODE3_IP> <NODE2_IP> <NODE4_IP> --dist non_iid
python3 node.py 4 <NODE4_IP> <NODE3_IP> <NODE5_IP> --dist non_iid
python3 node.py 5 <NODE5_IP> <NODE4_IP> <NODE6_IP> --dist non_iid
python3 node.py 6 <NODE6_IP> <NODE5_IP> <NODE7_IP> --dist non_iid
python3 node.py 7 <NODE7_IP> <NODE6_IP> <NODE0_IP> --dist non_iid
```

---

### Experiment 5-A — Decentralized | Fault Tolerance Demo (No SPOF)

Node 3 will automatically exit at round 16. Nodes 2 and 4 (its neighbors) detect the push failure, log it, and continue. All other nodes finish all 30 rounds unaffected.

```bash
python3 node.py 0 <NODE0_IP> <NODE7_IP> <NODE1_IP> --dist non_iid --fault-demo
python3 node.py 1 <NODE1_IP> <NODE0_IP> <NODE2_IP> --dist non_iid --fault-demo
python3 node.py 2 <NODE2_IP> <NODE1_IP> <NODE3_IP> --dist non_iid --fault-demo
python3 node.py 3 <NODE3_IP> <NODE2_IP> <NODE4_IP> --dist non_iid --fault-demo   # exits at round 16
python3 node.py 4 <NODE4_IP> <NODE3_IP> <NODE5_IP> --dist non_iid --fault-demo
python3 node.py 5 <NODE5_IP> <NODE4_IP> <NODE6_IP> --dist non_iid --fault-demo
python3 node.py 6 <NODE6_IP> <NODE5_IP> <NODE7_IP> --dist non_iid --fault-demo
python3 node.py 7 <NODE7_IP> <NODE6_IP> <NODE0_IP> --dist non_iid --fault-demo
```

**Expected outcome**: Nodes 2 and 4 log `[NEIGHBOR DOWN]` from round 16 onward. All 7 remaining nodes complete round 30.

---

### Experiment 5-B — Centralized | SPOF Demo

The server will automatically exit at round 16. All 7 clients will stall, log `SPOF CONFIRMED`, and cannot proceed because there is no alternative server.

**Node 0 (server):**
```bash
python3 server.py <NODE0_PRIVATE_IP> --fault-demo
```

**Nodes 1-7 (clients):**
```bash
python3 client.py 1 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 2 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 3 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 4 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 5 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 6 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
python3 client.py 7 <NODE0_PRIVATE_IP> --dist non_iid --fault-demo
```

**Expected outcome**: Server exits silently at round 16. All 7 clients print `CRITICAL -- SERVER UNREACHABLE` and `SPOF CONFIRMED` then stop.

---

## What to Collect Per Experiment

For each experiment, save the terminal output from every node. Key metrics to extract:

- Per-round accuracy (local and post-aggregation/post-blend)
- Communication time per round
- Total bytes exchanged
- Rounds completed (especially for Exp 5-A and 5-B)

The final summary table is printed automatically at the end of each run.

---

## Cost Notes (AWS)

- **t3.micro** is Free Tier eligible (750 hours/month per account for 12 months)
- All inter-node communication uses **private IPs** — no data transfer charges within the same region
- **Stop** (not terminate) instances between sessions to avoid charges while not running
- **Terminate all instances** after thesis experiments are fully complete

---

## Acknowledgements

- Base gossip FL concept: inspired by the GitFL paper on asynchronous gossip-based federated learning
- Dataset: CIFAR-10 (Krizhevsky, 2009)
- Model architecture: standard CNNCifar used in FL literature
