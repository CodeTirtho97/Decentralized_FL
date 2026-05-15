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
This repository benchmarks two FL designs under identical training settings:

- Centralized FL: one server aggregates client models (FedAvg).
- Decentralized FL (Ring): gossip where each node exchanges with left and right neighbors only.
- Decentralized FL (Fully Connected): gossip where each node exchanges with all other nodes per round.

The comparison covers:
- IID vs Non-IID data distributions
- Communication behavior and overhead
- Effect of graph topology on convergence (ring vs fully connected)
- Fault tolerance (SPOF vs no-SPOF demonstration)

## Key Features
- End-to-end scripts for 5 thesis experiments
- Uniform codebase for centralized and decentralized pipelines
- Structured per-node logging by date and experiment
- Fault-tolerance demo scenarios:
  - Decentralized: one node fails, system continues
  - Centralized: server fails, clients halt

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

## Project Structure
```text
VM_Decentralized/
|-- server.py                  # Centralized server
|-- client.py                  # Centralized client
|-- node.py                    # Decentralized node (ring topology)
|-- node_fc.py                 # Decentralized node (fully-connected topology)
|-- exp1_centralized_iid.sh
|-- exp2_centralized_noniid.sh
|-- exp3_decentralized_iid.sh
|-- exp4_decentralized_noniid.sh
|-- exp5a_decentralized_fault.sh
|-- exp5b_centralized_spof.sh
|-- exp6a_decentralized_fc_iid.sh
|-- exp6b_decentralized_fc_noniid.sh
|-- exp6c_decentralized_fc_fault.sh
|-- shared/
|   |-- data.py                # CIFAR-10 loading + IID/Non-IID split
|   |-- model.py               # CNNCifar architecture
|   |-- train.py               # Local train, evaluate, FedAvg
|   |-- net.py                 # TCP send/recv utilities
|   |-- log.py                 # Console + file logging helpers
|   `-- __init__.py
|-- How to Run.md              # Full operational runbook
|-- aws_node_ips.md            # Node private/public IP reference
`-- results/                   # Collected output logs and reports
```

## Experimental Design
### Experiments
| ID | Scenario | Architecture | Topology | Distribution |
|---|---|---|---|---|
| 1 | Baseline | Centralized | Star (server–client) | IID |
| 2 | Heterogeneous data | Centralized | Star (server–client) | Non-IID (Dirichlet alpha=0.5) |
| 3 | Baseline without server | Decentralized | Ring (2 neighbors) | IID |
| 4 | Heterogeneous data without server | Decentralized | Ring (2 neighbors) | Non-IID (Dirichlet alpha=0.5) |
| 5-A | Fault tolerance demo | Decentralized | Ring (2 neighbors) | Non-IID |
| 5-B | SPOF demo | Centralized | Star (server–client) | Non-IID |
| 6-A | Topology comparison — upper bound | Decentralized | Fully Connected (7 neighbors) | IID |
| 6-B | Topology comparison — upper bound | Decentralized | Fully Connected (7 neighbors) | Non-IID (Dirichlet alpha=0.5) |
| 6-C | FC fault tolerance demo | Decentralized | Fully Connected (7 neighbors) | Non-IID (Dirichlet alpha=0.5) |

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
- Packages:
  - `torch`
  - `torchvision`
  - `numpy`
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

Pre-download CIFAR-10 once:

```bash
cd ~/fl_project
source venv/bin/activate
python3 -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True); torchvision.datasets.CIFAR10(root='./data', train=False, download=True); print('CIFAR-10 ready')"
```

## Run Experiments
Use the shell wrappers from each node:

```bash
./exp<N>_<name>.sh <node_id>
```

Examples:

```bash
./exp1_centralized_iid.sh 0
./exp1_centralized_iid.sh 1

./exp3_decentralized_iid.sh 0
./exp3_decentralized_iid.sh 1
```

Detailed run order, SSH commands, and log download commands are documented in:
- `How to Run.md`

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

Local collected logs and summaries are kept in this repository under `results/`.

## Reproducibility Notes
- Random seeds are set in `shared/data.py` for dataset partition determinism.
- Private IPs should be used for node-to-node FL traffic.
- Public IPs are for SSH/SCP only and can change after EC2 restart.
- Keep all nodes in same VPC/subnet/security group for stable communication.

## Known Limitations
- In Non-IID mode, the current Dirichlet split can produce fewer than 6,250 samples for some nodes depending on class allocation. This can affect strict apples-to-apples fairness if fixed sample count is required.

## Troubleshooting
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
