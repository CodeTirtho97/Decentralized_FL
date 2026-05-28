# Log Format Reference

Every log file contains one `[COMM_SUMMARY]` line at the end of each round,
emitted by all scripts (`server.py`, `client.py`, `node.py`, `node_fc.py`,
`fedstellar_node.py`). These lines are what `tools/parse_comm_bottleneck.py`
parses to generate the thesis comparison table.

---

## Format

```
[COMM_SUMMARY] round=X arch=<arch> node=<id|server>
  train_time=XX.XXs comm_time=XX.XXs round_time=XX.XXs
  bytes_sent=XXXXX bytes_recv=XXXXX
  push_retries=N timeout_hits=N neighbors_received=N fanout=N
  pre_blend_acc=XX.XX% post_blend_acc=XX.XX%
```

## Field Definitions

| Field | Meaning |
|-------|---------|
| `arch` | `centralized`, `ring`, `fc`, or `fedstellar_ring` |
| `node` | Node ID (integer) or `server` |
| `train_time` | Local SGD training duration this round |
| `comm_time` | Network wait time: upload+download (centralized), push+receive (gossip), gossip+aggregate (Fedstellar) |
| `round_time` | `train_time + comm_time` |
| `bytes_sent` | Bytes pushed out this round |
| `bytes_recv` | Bytes received this round |
| `push_retries` | Extra reconnect attempts beyond first try |
| `timeout_hits` | Push operations that exhausted the full retry deadline |
| `neighbors_received` | Number of neighbor/peer models actually received |
| `fanout` | Number of neighbors this node expected to exchange with |
| `pre_blend_acc` | Test accuracy after local training, before blending (`N/A` for server) |
| `post_blend_acc` | Test accuracy after receiving and blending neighbor/aggregated models (`N/A` for server) |

---

## Log File Structure

Logs are organised by **date → experiment → node**.

```
logs/
  YYYY_MM_DD/
    centralized_iid/
      server.log
      client_1.log  ...  client_N.log
    centralized_noniid/
      server.log
      client_1.log  ...  client_N.log
    decentralized_iid/
      node_0.log  ...  node_N.log
    decentralized_noniid/
      node_0.log  ...  node_N.log
    centralized_spof/
      server.log
      client_1.log  ...  client_N.log
    decentralized_fault/
      node_0.log  ...  node_N.log
    decentralized_fc_iid/
      node_0.log  ...  node_N.log
    decentralized_fc_noniid/
      node_0.log  ...  node_N.log
    decentralized_fc_fault/
      node_0.log  ...  node_N.log
    fedstellar_ring_iid/
      node_0.log  ...  node_N.log
    fedstellar_ring_fault/
      node_0.log  ...  node_N.log
```

Re-running the same experiment on the same day overwrites the previous file.

---

## Parsing

```bash
# Single experiment
python tools/parse_comm_bottleneck.py --log_dir results/YYYY_MM_DD/decentralized_iid

# Compare two architectures side by side
python tools/parse_comm_bottleneck.py --log_dir results/YYYY_MM_DD --arch ring centralized

# Scan an entire date folder — all experiments at once
python tools/parse_comm_bottleneck.py --log_dir results/YYYY_MM_DD

# Filter to specific architectures
python tools/parse_comm_bottleneck.py --log_dir results/YYYY_MM_DD --arch ring fc
```

If the parser reports `No [COMM_SUMMARY] lines found`, the experiment was run
with old scripts that did not emit these lines.
