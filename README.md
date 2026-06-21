# Federated Learning Thesis: Centralized vs Decentralized (AWS, 8 Nodes)

A thesis implementation that compares centralized and decentralized federated learning (FL) on CIFAR-10 across 8 AWS EC2 instances.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Experimental Design](#experimental-design)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Run Experiments](#run-experiments)
- [Logging and Results](#logging-and-results)
- [Reproducibility Notes](#reproducibility-notes)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)

## Overview
This repository benchmarks FL designs under identical training settings:

- Centralized FL: one server aggregates client models (FedAvg).
- Decentralized FL (Ring): gossip where each node exchanges with left and right neighbors only.
- Decentralized FL (Fully Connected): gossip where each node exchanges with all other nodes per round.
- Fedstellar (p2pfl): same ring topology using the p2pfl async gRPC platform instead of hand-rolled TCP.

The comparison covers:
- IID vs Non-IID data distributions
- Communication behavior and overhead
- Effect of graph topology on convergence (ring vs fully connected)
- Fault tolerance (SPOF vs no-SPOF demonstration)
- Effect of synchronization model on fault cascade severity (synchronous TCP vs async gRPC)

## Key Features
- End-to-end scripts for 11 thesis experiments
- Uniform codebase for centralized, decentralized, and Fedstellar (p2pfl) pipelines
- Structured per-node logging by date and experiment
- Machine-readable `[COMM_SUMMARY]` lines in every log for bottleneck analysis
- Fault-tolerance demo scenarios:
  - Decentralized (synchronous): one node fails, neighbors detect 90+120s TCP timeout cascade
  - Centralized: server fails, all clients halt (SPOF)
  - Fedstellar (async gRPC): same fault injected to compare cascade behavior across synchronization models

## System Architecture
### Centralized FL (Experiments 1, 2, 5-B)
- Node 0 runs `server.py`
- Nodes 1-7 run `client.py`
- Upload port: `9000`
- Broadcast port: `9001`

### Decentralized FL — Ring (Experiments 3, 4, 5-A)
- All 8 nodes run `node.py`
- Ring topology: each node communicates with 2 neighbors only
- Listener port per node: `8000 + node_id`
- Synchronous round pattern: `train -> push -> receive -> blend -> evaluate`

### Decentralized FL — Fully Connected (Experiments 6-A, 6-B, 6-C)
- All 8 nodes run `node_fc.py`
- Fully-connected topology: each node communicates with all other 7 nodes per round
- Same port scheme as ring: listener port = `8000 + node_id`
- Same synchronous pattern: `train -> push to all -> receive from all -> blend -> evaluate`
- Blend pool: 8 models (own + 7 peers) vs 3 in ring (own + 2 neighbors)
- Exp 6-C: fault demo — Node 3 exits at round 10; all 7 surviving nodes detect the failure (vs only 2 in ring Exp 5-A)

### Fedstellar — p2pfl Ring (Experiments 7, 7-F)
- All 8 nodes run `fedstellar_experiment/fedstellar_node.py` via the p2pfl library
- Same ring topology as Experiments 3 and 5-A (Node i connects to (i-1)%8 and (i+1)%8)
- Transport: gRPC (async) instead of raw TCP (synchronous blocking)
- p2pfl listener port per node: `10000 + node_id` (avoids clash with ring experiments)
- Exp 7: IID accuracy comparison vs Exp 3 (same data, same model, different FL framework)
- Exp 7-F: fault demo — Node 3 exits at round 10; measures whether async gRPC suppresses the synchronous TCP cascade from Exp 5-A
- Only IID data used: Non-IID excluded to isolate protocol effect from data heterogeneity effect

## Project Structure
```text
VM_Decentralized/
|-- server.py                  # Centralized server
|-- client.py                  # Centralized client
|-- node.py                    # Decentralized node (ring topology, synchronous TCP)
|-- node_fc.py                 # Decentralized node (fully-connected topology, synchronous TCP)
|-- node_timeout_{high,mid,low}.py  # Ring node variants for the optional timeout study (90/120, 40/60, 15/15)
|-- exp1_centralized_iid.sh
|-- exp2_centralized_noniid.sh
|-- exp3_decentralized_iid.sh
|-- exp4_decentralized_noniid.sh
|-- exp5a_decentralized_fault.sh
|-- exp5b_centralized_spof.sh
|-- exp6a_decentralized_fc_iid.sh
|-- exp6b_decentralized_fc_noniid.sh
|-- exp6c_decentralized_fc_fault.sh
|-- exp5a_timeout_{high,mid,low}.sh  # Optional timeout-sensitivity study (run per-node manually)
|-- launch.sh                  # One-command launcher for Exps 1-6 (all nodes via SSH)
|-- launch_fedstellar.sh       # One-command launcher for Exps 7, 7-F (all nodes via SSH)
|-- shared/
|   |-- data.py                # CIFAR-10 loading + IID/Non-IID split
|   |-- model.py               # CNNCifar architecture
|   |-- train.py               # Local train, evaluate, FedAvg
|   |-- net.py                 # TCP send/recv utilities
|   |-- log.py                 # Console + file logging helpers
|   `-- __init__.py
|-- fedstellar_experiment/     # Experiments 7 and 7-F (p2pfl / Fedstellar platform)
|   |-- fedstellar_node.py     # p2pfl node: ring, IID, [COMM_SUMMARY] logging
|   |-- exp_fedstellar_iid.sh  # Per-node launch script (Exp 7)
|   |-- exp_fedstellar_fault.sh# Per-node launch script (Exp 7-F, Node 3 exits at round 10)
|   |-- setup_fedstellar.sh    # One-time pip install: p2pfl, lightning, grpcio
|   `-- shared/
|       |-- model_lightning.py # CNNCifar as LightningModule (identical architecture)
|       `-- data_fedstellar.py # CIFAR-10 IID partition for p2pfl
|-- tools/
|   `-- parse_comm_bottleneck.py  # Parse [COMM_SUMMARY] logs → markdown comparison table
|-- How_To_Implement.md        # Full replication guide (all 11 experiments)
|-- aws_node_ips.md            # Node private/public IP reference
`-- results/                   # Collected output logs and reports
```

## Experimental Design
### Experiments
| ID | Scenario | Architecture | Topology | Distribution |
|---|---|---|---|---|
| 1 | Baseline | Centralized | Star (server–client) | IID |
| 2 | Heterogeneous data | Centralized | Star (server–client) | Non-IID (Dirichlet alpha=0.5) |
| 3 | Baseline without server | Decentralized (sync TCP) | Ring (2 neighbors) | IID |
| 4 | Heterogeneous data without server | Decentralized (sync TCP) | Ring (2 neighbors) | Non-IID (Dirichlet alpha=0.5) |
| 5-A | Fault tolerance demo | Decentralized (sync TCP) | Ring (2 neighbors) | Non-IID |
| 5-B | SPOF demo | Centralized | Star (server–client) | Non-IID |
| 6-A | Topology comparison — upper bound | Decentralized (sync TCP) | Fully Connected (7 neighbors) | IID |
| 6-B | Topology comparison — upper bound | Decentralized (sync TCP) | Fully Connected (7 neighbors) | Non-IID (Dirichlet alpha=0.5) |
| 6-C | FC fault tolerance demo | Decentralized (sync TCP) | Fully Connected (7 neighbors) | Non-IID (Dirichlet alpha=0.5) |
| 7 | Framework comparison (IID) | Fedstellar (p2pfl async gRPC) | Ring (2 neighbors) | IID |
| 7-F | Fault cascade comparison | Fedstellar (p2pfl async gRPC) | Ring (2 neighbors) | IID |

> **Why IID only for Experiments 7 and 7-F?** The goal is to isolate the effect of the FL communication protocol (synchronous TCP blocking vs. asynchronous gRPC). Adding Non-IID would introduce a second independent variable (data heterogeneity) that would confound attribution of any accuracy difference to the framework. IID gives a clean baseline already validated by Exp 3. Exp 7-F then tests whether the TCP cascade from Exp 5-A disappears under async gRPC with all other variables held fixed.

> **Optional timeout-sensitivity study.** The `exp5a_timeout_{high,mid,low}.sh` scripts re-run the ring fault demo (Exp 5-A) at three push/receive timeout settings (90/120 s, 40/60 s, 15/15 s) to measure how timeout values affect fault-cascade severity. Run per-node manually — see [Appendix A in How_To_Implement.md](How_To_Implement.md#appendix-a-optional-timeout-sensitivity-study).

### Default Training Parameters
| Parameter | Value |
|---|---|
| FL rounds | 50 |
| Local epochs per round | 5 |
| Batch size | 64 |
| Target samples per node | 6,250 |
| Optimizer | SGD (lr=0.01, momentum=0.5) |
| Non-IID Dirichlet alpha | 0.5 |
| Inter-round sleep | 5 seconds |
| Fault trigger round | 10 |

## Requirements
- OS: Ubuntu 24.04 LTS (AWS EC2 nodes)
- Python: 3.10+
- Runtime: CPU-only PyTorch
- Packages (Experiments 1–6):
  - `torch`
  - `torchvision`
  - `numpy`
- Additional packages (Experiments 7, 7-F — Fedstellar):
  - `p2pfl==0.4.4`
  - `lightning>=2.0.0`
  - `grpcio>=1.54.0`
  - `grpcio-tools>=1.54.0`
- Tools:
  - `tmux`
  - `scp`

## Environment Setup
Run on each EC2 node:

```bash
sudo apt update -y
sudo apt install -y python3-pip python3-venv tmux

mkdir -p ~/fl_project/data
cd ~/fl_project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy --index-url https://download.pytorch.org/whl/cpu
```

Add swap (required on t3.micro — prevents OOM when all 8 nodes start simultaneously):

```bash
sudo fallocate -l 2G /swapfile; sudo chmod 600 /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile
```

> Run this after every EC2 restart. Swap is not persistent across reboots on these instances.

Pre-download CIFAR-10 once:

```bash
cd ~/fl_project
source venv/bin/activate
python3 -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True); torchvision.datasets.CIFAR10(root='./data', train=False, download=True); print('CIFAR-10 ready')"
```

## Run Experiments

### Experiments 1–6 (hand-rolled TCP gossip)
Use `launch.sh` from your laptop (Git Bash / WSL):
```bash
./launch.sh 3    # Decentralized ring IID (Exp 3)
./launch.sh 5a   # Decentralized fault tolerance (Exp 5-A)
```
Or run per-node scripts directly on each EC2 instance:
```bash
./exp3_decentralized_iid.sh <node_id>
```

### Experiments 7 and 7-F (Fedstellar / p2pfl)
First-time setup (one-time per node):
```bash
./launch_fedstellar.sh upload   # sync fedstellar_experiment/ to all nodes
./launch_fedstellar.sh setup    # pip install p2pfl + lightning on all nodes
```
After every EC2 restart, add swap before launching (p2pfl needs ~700 MB per node; t3.micro OOMs without swap). One command does all 8 nodes (safe to re-run):
```bash
./launch_fedstellar.sh swap
```
Then launch:
```bash
./launch_fedstellar.sh iid      # Exp 7: Fedstellar ring IID
./launch_fedstellar.sh fault    # Exp 7-F: Fedstellar ring fault demo
```

Detailed run order, SSH commands, and log download commands are documented in:
- [How_To_Implement.md](How_To_Implement.md)

## Logging and Results
Each run writes logs to:

```text
~/fl_project/logs/YYYY_MM_DD/<experiment_label>/
```

Typical labels:
- `centralized_iid`
- `centralized_noniid`
- `decentralized_iid`
- `decentralized_noniid`
- `decentralized_fault`
- `centralized_spof`
- `decentralized_fc_iid`
- `decentralized_fc_noniid`
- `decentralized_fc_fault`
- `fedstellar_ring_iid`        ← Exp 7 (p2pfl)
- `fedstellar_ring_fault`      ← Exp 7-F (p2pfl fault demo)

Local collected logs and summaries are kept in this repository under `results/`.

## Reproducibility Notes
- Random seeds are set in `shared/data.py` for dataset partition determinism.
- Private IPs should be used for node-to-node FL traffic.
- Public IPs are for SSH/SCP only and can change after EC2 restart.
- Keep all nodes in same VPC/subnet/security group for stable communication.

## Known Limitations
- In Non-IID mode, the current Dirichlet split can produce fewer than 6,250 samples for some nodes depending on class allocation. This can affect strict apples-to-apples fairness if fixed sample count is required.

## Troubleshooting
- If Fedstellar (Exp 7 / 7-F) nodes crash immediately after launch:
  - t3.micro runs out of RAM when all 8 nodes start simultaneously.
  - Add swap before launching (see "Run Experiments" above). Each node needs ~1.6 GB swap.
  - If disk is full (`df -h /` shows 100%), clean pip cache first: `sudo rm -rf ~/.cache/pip /root/.cache/pip`
- If a run seems stuck:
  - Verify all 8 nodes are reachable and running.
  - Confirm security group allows intra-group TCP traffic.
  - Confirm correct private IP mapping for node IDs.
- If SSH disconnects during runs:
  - Reattach tmux session:

```bash
tmux attach -t exp
```

- If CIFAR-10 is missing:
  - Re-run the pre-download command in setup.

## Security Notes
- Do not commit private keys (`*.pem`) to public repositories.
- Use private IPs for FL communication whenever possible.
- Restrict inbound SSH (port 22) to trusted IP ranges where feasible.

## License
Add your project license here (for example, MIT, Apache-2.0, or thesis-specific academic use terms).
