#!/bin/bash
# =============================================================================
#  run_fedstellar.sh  --  Launch Fedstellar (p2pfl) experiment on this node
#
#  Reads node IPs from config.env in the project root.
#  Run on ALL nodes simultaneously (start within 30 seconds of each other).
#
#  Usage:
#    bash experiments/fedstellar/run_fedstellar.sh --exp iid    --node-id <id>
#    bash experiments/fedstellar/run_fedstellar.sh --exp fault  --node-id <id>
#
#  Prerequisites (one-time per node):
#    bash experiments/fedstellar/setup_fedstellar.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if [[ ! -f config.env ]]; then
    echo "  ERROR: config.env not found. Copy config.env.example and fill in your IPs."
    exit 1
fi
source config.env

EXP=""
NODE_ID=""
ROUNDS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)      EXP="$2";     shift 2 ;;
        --node-id)  NODE_ID="$2"; shift 2 ;;
        --rounds)   ROUNDS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXP" || -z "$NODE_ID" ]]; then
    echo ""
    echo "  Usage: bash experiments/fedstellar/run_fedstellar.sh --exp iid|fault --node-id <id>"
    echo ""
    exit 1
fi

MY_IP="${NODE_IPS[$NODE_ID]}"
LEFT_ID=$(( (NODE_ID - 1 + NUM_NODES) % NUM_NODES ))
RIGHT_ID=$(( (NODE_ID + 1) % NUM_NODES ))
LEFT_IP="${NODE_IPS[$LEFT_ID]}"
RIGHT_IP="${NODE_IPS[$RIGHT_ID]}"

SESSION="fedstellar_${EXP}_node${NODE_ID}"
if [[ -z "$TMUX" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    exec tmux new-session -s "$SESSION" \
        "cd '$PROJECT_ROOT' && bash experiments/fedstellar/run_fedstellar.sh --exp $EXP --node-id $NODE_ID --rounds $ROUNDS"
fi

if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi
export PYTHONPATH="$PROJECT_ROOT/src"

echo ""
echo "  Fedstellar experiment : $EXP"
echo "  Node ID               : $NODE_ID  ($MY_IP)"
echo "  Left neighbor         : Node $LEFT_ID  ($LEFT_IP)"
echo "  Right neighbor        : Node $RIGHT_ID  ($RIGHT_IP)"
echo ""

CRASH_LOG="$PROJECT_ROOT/logs/fedstellar_crash_node_${NODE_ID}.txt"
mkdir -p "$PROJECT_ROOT/logs"
echo "=== $(date) ===" > "$CRASH_LOG"

FAULT_FLAG=""
if [[ "$EXP" == "fault" ]]; then
    FAULT_FLAG="--fault-demo"
fi

python3 -m experiments.fedstellar.fedstellar_node \
    "$NODE_ID" "$MY_IP" "$LEFT_IP" "$RIGHT_IP" \
    --rounds "$ROUNDS" $FAULT_FLAG \
    2>&1 | tee -a "$CRASH_LOG"

echo "=== EXIT $(date) ===" >> "$CRASH_LOG"
