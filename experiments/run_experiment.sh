#!/bin/bash
# =============================================================================
#  run_experiment.sh  --  Launch any experiment on this node
#
#  Reads node IPs and settings from config.env in the project root.
#  Run on EVERY node simultaneously (or Node 0 first for centralized).
#
#  Usage:
#    bash experiments/run_experiment.sh --exp <experiment> --node-id <id>
#
#  Available experiments:
#    centralized_iid          Exp 1: Centralized FedAvg, IID data
#    centralized_noniid       Exp 2: Centralized FedAvg, Non-IID data
#    decentralized_iid        Exp 3: Ring gossip, IID data
#    decentralized_noniid     Exp 4: Ring gossip, Non-IID data
#    decentralized_fault      Exp 5-A: Ring gossip, Node 3 exits at round 10
#    centralized_spof         Exp 5-B: Centralized, server exits at round 10
#    decentralized_fc_iid     Exp 6-A: Fully-connected gossip, IID data
#    decentralized_fc_noniid  Exp 6-B: Fully-connected gossip, Non-IID data
#    decentralized_fc_fault   Exp 6-C: Fully-connected, Node 3 exits at round 10
#
#  Examples:
#    bash experiments/run_experiment.sh --exp centralized_iid  --node-id 0
#    bash experiments/run_experiment.sh --exp decentralized_iid --node-id 3
#    bash experiments/run_experiment.sh --exp decentralized_fault --node-id 0 --rounds 50
# =============================================================================

set -e

# ---- Locate project root (two dirs up from this script) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- Load user configuration ----
if [[ ! -f config.env ]]; then
    echo ""
    echo "  ERROR: config.env not found."
    echo "  Copy config.env.example to config.env and fill in your node IPs."
    echo ""
    exit 1
fi
source config.env

# ---- Defaults (can be overridden by config.env or CLI flags) ----
ROUNDS=50
EPOCHS=5
BATCH=64
SAMPLES=6250
ALPHA=0.5

# ---- Parse CLI arguments ----
EXP=""
NODE_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)       EXP="$2";       shift 2 ;;
        --node-id)   NODE_ID="$2";   shift 2 ;;
        --rounds)    ROUNDS="$2";    shift 2 ;;
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --alpha)     ALPHA="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXP" || -z "$NODE_ID" ]]; then
    echo ""
    echo "  Usage: bash experiments/run_experiment.sh --exp <experiment> --node-id <id>"
    echo ""
    echo "  Experiments:"
    echo "    centralized_iid | centralized_noniid"
    echo "    decentralized_iid | decentralized_noniid | decentralized_fault | centralized_spof"
    echo "    decentralized_fc_iid | decentralized_fc_noniid | decentralized_fc_fault"
    echo ""
    exit 1
fi

# ---- Validate node ID ----
if [[ $NODE_ID -ge $NUM_NODES ]]; then
    echo "  ERROR: --node-id $NODE_ID is out of range (0 to $((NUM_NODES - 1)))"
    exit 1
fi

MY_IP="${NODE_IPS[$NODE_ID]}"

# ---- Auto-wrap in tmux (protects against SSH disconnect) ----
SESSION="fl_exp_${EXP}_node${NODE_ID}"
if [[ -z "$TMUX" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    exec tmux new-session -s "$SESSION" \
        "cd '$PROJECT_ROOT' && bash experiments/run_experiment.sh --exp $EXP --node-id $NODE_ID --rounds $ROUNDS $*"
fi

# ---- Activate venv ----
if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi
export PYTHONPATH="$PROJECT_ROOT/src"

echo ""
echo "  Experiment  : $EXP"
echo "  Node ID     : $NODE_ID  ($MY_IP)"
echo "  Rounds      : $ROUNDS"
echo ""

# ---- Ring topology helpers ----
LEFT_ID=$(( (NODE_ID - 1 + NUM_NODES) % NUM_NODES ))
RIGHT_ID=$(( (NODE_ID + 1) % NUM_NODES ))
LEFT_IP="${NODE_IPS[$LEFT_ID]}"
RIGHT_IP="${NODE_IPS[$RIGHT_ID]}"
SERVER_IP="${NODE_IPS[0]}"   # Node 0 is always the centralized server

# ---- Dispatch ----
case "$EXP" in

    centralized_iid)
        if [[ $NODE_ID -eq 0 ]]; then
            python3 src/server.py "$MY_IP" --clients $((NUM_NODES - 1)) --rounds $ROUNDS --dist iid
        else
            python3 src/client.py "$NODE_ID" "$SERVER_IP" --dist iid --rounds $ROUNDS --num-nodes $NUM_NODES
        fi
        ;;

    centralized_noniid)
        if [[ $NODE_ID -eq 0 ]]; then
            python3 src/server.py "$MY_IP" --clients $((NUM_NODES - 1)) --rounds $ROUNDS --dist non_iid
        else
            python3 src/client.py "$NODE_ID" "$SERVER_IP" --dist non_iid --alpha $ALPHA --rounds $ROUNDS --num-nodes $NUM_NODES
        fi
        ;;

    decentralized_iid)
        python3 src/node.py "$NODE_ID" "$MY_IP" "$LEFT_IP" "$RIGHT_IP" \
            --dist iid --rounds $ROUNDS --num-nodes $NUM_NODES
        ;;

    decentralized_noniid)
        python3 src/node.py "$NODE_ID" "$MY_IP" "$LEFT_IP" "$RIGHT_IP" \
            --dist non_iid --alpha $ALPHA --rounds $ROUNDS --num-nodes $NUM_NODES
        ;;

    decentralized_fault)
        python3 src/node.py "$NODE_ID" "$MY_IP" "$LEFT_IP" "$RIGHT_IP" \
            --dist non_iid --alpha $ALPHA --rounds $ROUNDS --num-nodes $NUM_NODES --fault-demo
        ;;

    centralized_spof)
        if [[ $NODE_ID -eq 0 ]]; then
            python3 src/server.py "$MY_IP" --clients $((NUM_NODES - 1)) --rounds $ROUNDS --dist non_iid --fault-demo
        else
            python3 src/client.py "$NODE_ID" "$SERVER_IP" --dist non_iid --alpha $ALPHA --rounds $ROUNDS --num-nodes $NUM_NODES --fault-demo
        fi
        ;;

    decentralized_fc_iid)
        python3 src/node_fc.py "$NODE_ID" "${NODE_IPS[@]}" \
            --dist iid --rounds $ROUNDS
        ;;

    decentralized_fc_noniid)
        python3 src/node_fc.py "$NODE_ID" "${NODE_IPS[@]}" \
            --dist non_iid --alpha $ALPHA --rounds $ROUNDS
        ;;

    decentralized_fc_fault)
        python3 src/node_fc.py "$NODE_ID" "${NODE_IPS[@]}" \
            --dist non_iid --alpha $ALPHA --rounds $ROUNDS --fault-demo
        ;;

    *)
        echo "  ERROR: Unknown experiment '$EXP'"
        echo "  Run with --help or no args to see available experiments."
        exit 1
        ;;
esac
