# Synchronization and Fault Propagation in Decentralized Federated Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%2022.04-e95420?logo=ubuntu&logoColor=white)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20EC2-ff9900?logo=amazonec2&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-6366f1)
![License](https://img.shields.io/badge/License-MIT-22c55e)

Federated Learning allows distributed devices to collaboratively train a model without
sharing raw data, but the dominant architecture — a central aggregation server — introduces
a structural fragility that is widely acknowledged yet rarely demonstrated on real
infrastructure. This thesis builds both a centralized and a decentralized peer-to-peer
gossip FL system from scratch, deploys them on 8 AWS EC2 instances using raw TCP sockets,
and subjects each to controlled failure injection under identical hardware and data
conditions. Beyond confirming fault tolerance differences, the experiments uncover a
**cascading synchronization failure** in the ring gossip protocol — an emergent behavior
where a single node's death eventually isolates its neighbors from the entire ring through
TCP timeout inflation — a dynamic that existing theoretical FL models do not predict.

---

## What This Is

This project implements and empirically compares two Federated Learning architectures
deployed on **8 real AWS EC2 instances**:

- **Centralized FL (FedAvg)** — McMahan et al. 2017. A central server aggregates models
  from all clients each round. Standard star topology.

- **Decentralized FL (Gossip)** — No server. Nodes train locally and exchange model weights
  peer-to-peer in a ring or fully-connected topology. Synchronous equal-weight FedAvg blend.

Both architectures were **implemented from scratch** in Python using raw TCP sockets —
no FL framework (Flower, PySyft, etc.) for Experiments 1–6. The same CNN model,
dataset (CIFAR-10), hyperparameters, and hardware were used for all experiments so
that topology is the only variable that changes between centralized and decentralized comparisons.

Experiment 7 deploys the same ring topology via the **Fedstellar (p2pfl)** framework
(gRPC-based gossip) to directly compare a framework-based implementation against the
hand-rolled synchronous TCP implementation under identical conditions.

---

## Why This Was Done

Federated Learning surveys and theory papers consistently argue that decentralized FL
eliminates the Single Point of Failure (SPOF) of centralized architectures. But this
argument is always made **structurally** — "there is no central server, therefore no SPOF."

No prior work:
1. **Empirically demonstrates** SPOF in centralized FL through live server failure injection
   on real distributed infrastructure (timestamped logs, real network conditions, real retries)
2. **Characterizes the secondary failure dynamics** that emerge when a ring node fails under
   synchronous TCP coupling — specifically, how a single node's death propagates through
   timeout inflation to eventually isolate its neighbors from the rest of the ring
3. **Quantifies the accuracy-fault tolerance tradeoff** under controlled, identical hardware
   conditions: same model, same dataset, same instance type, only topology changes

This thesis fills those gaps through **empirical systems research**: build it, run it under
failure conditions, measure what theory does not predict.

### Research Questions

1. Does decentralized synchronous gossip FL converge to comparable accuracy as centralized
   FedAvg over 50 rounds, under IID and Non-IID data distributions?
2. What is the per-round communication overhead of ring gossip vs centralized FedAvg on
   real cloud infrastructure — in data volume and wall-clock time?
3. When the central server fails, what happens? When one ring node fails, what happens?
   (Empirical proof via live failure injection — not theoretical argument)
4. Does data heterogeneity (Non-IID, Dirichlet α=0.5) affect convergence differently in
   centralized vs decentralized architectures, and why?
5. What secondary behavioral effects emerge in ring gossip under node failure — effects
   not predicted by existing theoretical models?

---

## Architecture

### Centralized — Star Topology (Experiments 1, 2, 5-B)

```
           ┌─────────┐
    ┌──────►  SERVER  ◄──────┐
    │      │  Node 0 │       │
    │      └────┬────┘       │
    │       broadcast        │
  upload        │          upload
    │           ▼             │
 Client 1   Client 2  ...  Client 7
```

Round structure: `train locally → upload to server → server FedAvg → broadcast back`

All 7 clients must upload before aggregation begins. If the server goes down,
every client stalls and cannot recover — there is no fallback.

---

### Decentralized — Ring Topology (Experiments 3, 4, 5-A)

```
  0 ── 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── 7 ── (wraps to 0)
```

Round structure: `train locally → push to left + right neighbor → receive from neighbors → blend → evaluate`

No coordinator. Each node blends 3 models per round (self + 2 neighbors).
If one node goes down, its two direct neighbors detect the failure and continue
with 1 neighbor instead of 2. All other nodes are unaffected.

---

### Decentralized — Fully Connected Topology (Experiments 6-A, 6-B, 6-C)

```
  Every node connects to all 7 other nodes each round.
```

Round structure: `train locally → push to all 7 peers → receive from all 7 peers → blend 8 models → evaluate`

Theoretical upper bound for decentralized connectivity. Higher accuracy per round
than ring, but 7× the communication volume.

---

### Fedstellar — p2pfl Ring (Experiments 7, 7-F)

Same ring topology as Experiments 3 and 5-A, but using **p2pfl 0.4.4** (gRPC-based
semi-synchronous gossip) instead of hand-rolled synchronous TCP. Same model, same
data split, same number of rounds — only the communication framework changes.

---

## Project Structure

```
Decentralized_FL/
│
├── config.env.example      ← fill in your IPs, save as config.env (gitignored)
├── requirements.txt        ← pip dependencies
├── setup_nodes.sh          ← one-time node environment setup
│
├── src/                    ← all Python source (run with PYTHONPATH=src)
│   ├── server.py           ← centralized FL server (Exps 1, 2, 5-B)
│   ├── client.py           ← centralized FL client (Exps 1, 2, 5-B)
│   ├── node.py             ← decentralized ring gossip node (Exps 3, 4, 5-A)
│   ├── node_fc.py          ← decentralized fully-connected node (Exps 6-A/B/C)
│   └── shared/
│       ├── data.py         ← CIFAR-10 IID / Non-IID (Dirichlet) splits
│       ├── model.py        ← CNNCifar architecture (~62,006 parameters)
│       ├── net.py          ← length-prefixed TCP socket protocol
│       ├── log.py          ← stdout tee to terminal + log file
│       └── train.py        ← local SGD training, evaluation, FedAvg
│
├── experiments/
│   ├── run_experiment.sh   ← single launcher for all 9 experiments (reads config.env)
│   └── fedstellar/         ← p2pfl-based Fedstellar experiments
│       ├── fedstellar_node.py
│       ├── run_fedstellar.sh
│       ├── setup_fedstellar.sh
│       └── shared/         ← Lightning model wrapper, CIFAR-10 for p2pfl
│
├── tools/
│   ├── parse_comm_bottleneck.py       ← parses [COMM_SUMMARY] log lines → markdown table
│   └── generate_round_delay_graphs.py ← generates PNG round-delay graphs from logs
│
└── docs/
    ├── Implementation_Approach.md  ← research design rationale and defense guide
    └── LOG_FORMAT.md               ← [COMM_SUMMARY] line format specification
```

**Setup and operational instructions:** See [setup_nodes.sh](setup_nodes.sh) and
[docs/Implementation_Approach.md](docs/Implementation_Approach.md).

**Experiment launch commands and start order:** See [experiments/run_experiment.sh](experiments/run_experiment.sh).

**Log format and analysis commands:** See [docs/LOG_FORMAT.md](docs/LOG_FORMAT.md).

---

## Model

**CNNCifar** — lightweight CNN for CIFAR-10. Identical architecture across all experiments.

```
Input (3 × 32 × 32)
  → Conv2d(3→6, k=5) → ReLU → MaxPool2d(2×2)
  → Conv2d(6→16, k=5) → ReLU → MaxPool2d(2×2)
  → Flatten → Linear(400→120) → ReLU
  → Linear(120→84) → ReLU → Linear(84→10)

Parameters  : ~62,006
Model size  : ~245 KB (serialized)
Optimizer   : SGD  lr=0.01  momentum=0.5
```

---

## Results

All experiments: 8 AWS EC2 t3.micro instances (us-east-1), 50 FL rounds,
5 local epochs per round, CIFAR-10 (6,250 samples per node), seed=42.
Experiments 1–4 replicated 3× (May 7, 8, 9) — results consistent within 0.33 pp.

### Accuracy Comparison

| Experiment | Topology | Distribution | Peak Accuracy | Final Accuracy (R50) |
|------------|----------|--------------|:-------------:|:--------------------:|
| Exp 1 | Centralized | IID | **61.69%** (R28) | 60.02% |
| Exp 2 | Centralized | Non-IID | 57.60% (R26) | 56.58% |
| Exp 3 | Ring Gossip | IID | 57.48% (R21) | 55.98% |
| Exp 4 | Ring Gossip | Non-IID | 53.44% (R47) | 49.49% |
| Exp 6-A | Fully Connected | IID | 62.40% | 61.17% |
| Exp 6-B | Fully Connected | Non-IID | — | 57.99% |

**IID accuracy gap (centralized − ring gossip): −4.21 pp**
**Non-IID accuracy gap: −6.49 pp** (gossip degrades more due to smaller aggregation pool — 2 neighbors vs 7 clients)

---

### Communication Overhead

| Metric | Centralized (Exp 1) | Ring Gossip (Exp 3) | FC Gossip (Exp 6-A) |
|--------|:-------------------:|:-------------------:|:-------------------:|
| Avg comm time / round | 3.9 s | 25.9 s | 3.4 s |
| Avg round duration | 16.1 s | 37.5 s | 15.4 s |
| Data / round / node | 490.6 KB | 981.2 KB | 1717.1 KB |
| Total data / node (50R) | 24.5 MB | 49.1 MB | 167.7 MB |

**Bottleneck analysis:**
- **Centralized:** bottleneck is the server serializing N models (O(N) scaling). Comm time ~3.9s.
- **Ring gossip:** bottleneck is *waiting for the slower neighbor's training to complete* — not bandwidth. Comm time ~25.9s (6.6× longer than centralized), but would halve on homogeneous hardware with faster CPUs.
- **Fully connected:** same per-round time as centralized despite 7× data volume, because all 7 pushes happen in parallel threads.

---

### Fault Tolerance

| | Centralized SPOF (Exp 5-B) | Ring Fault (Exp 5-A) | FC Fault (Exp 6-C) |
|--|--|--|--|
| Failure event | Server exits after R9 | Node 3 exits at R10 | Node 3 exits at R10 |
| Nodes detecting failure | All 7 clients | 2 (direct ring neighbors) | 7 (all survivors) |
| System continues? | **No — permanent halt** | **Yes** | **Yes** |
| Rounds completed | 9 / 50 (0 of 7 clients complete) | 50 / 50 (7 of 8 nodes) | 50 / 50 (7 of 8 nodes) |
| Recovery mechanism | None — no fallback server | Self-healing: blends 1 model instead of 2 | Self-healing: blends 6 models instead of 7 |

---

### Cascading TCP Timeout Discovery

When Node 3 fails in the ring experiment (Exp 5-A), the 120-second TCP receive timeout
inflates the round duration of direct neighbors (Nodes 2 and 4) from ~37s to ~130s per round.
This timing inflation then cascades: the remaining neighbor of Node 2 and Node 4 cannot
synchronize with them, eventually causing Node 2 (round 38) and Node 4 (round 41) to lose
**ALL** connectivity — not just to Node 3, but to every other node in the ring.

This cascading isolation effect was **not designed** into the experiment. It emerged from
real distributed system behavior and is not predicted by theoretical gossip FL models.

Distance-dependent communication time inflation after Node 3 failure:

| Ring distance from Node 3 | Avg comm time R11–50 |
|:--------------------------|:--------------------:|
| Direct neighbors (Nodes 2, 4) | ~96.5 s / round |
| 1-hop (Nodes 1, 5) | ~65 s / round |
| 2-hops (Nodes 0, 6) | ~51 s / round |
| 3-hops (Node 7) | ~48.6 s / round |

---

## Limitations

These limitations are acknowledged as deliberate scope boundaries, not oversights.

- **Small scale (8 nodes).** Ring gossip convergence is O(N/2) hops — at 100 nodes, full model diffusion takes 50 rounds alone. Results are valid as a proof-of-concept; scaling behavior at larger N is left as future work.
- **Homogeneous hardware.** All nodes are identical t3.micro instances running synchronously. Heterogeneous devices with straggler nodes (different training speeds) represent a separate research problem — async FL — and are intentionally out of scope for a clean architectural comparison.
- **Fixed 120s receive timeout.** The cascading isolation effect (Nodes 2 and 4 losing all connectivity by rounds 38 and 41) is a direct consequence of this fixed timeout. Adaptive timeout backoff or skip-and-continue logic would prevent secondary isolation and is the most actionable direction for follow-up work.
- **Clean failure only.** Experiments inject honest node death — a node stops responding. Byzantine behavior (a malicious node sending corrupted weights) is a separate research track and is not tested here.
- **Synchronous protocol only.** Both architectures use synchronous round execution by design, to keep topology as the sole variable. Asynchronous gossip (e.g. Fedstellar) introduces staleness as a second variable and would make the centralized vs decentralized comparison uninterpretable.

---

## Acknowledgements

This work was carried out under the guidance of **Dr. Anshu S. Anand**, at the
**Indian Institute of Information Technology Allahabad** as part of the M.Tech
programme in Information Technology.

The experiments were conducted on AWS EC2 instances. CIFAR-10 dataset was used
solely for academic research purposes.
