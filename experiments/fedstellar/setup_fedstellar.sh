#!/bin/bash
# =============================================================================
#  setup_fedstellar.sh  --  Install p2pfl (Fedstellar) and dependencies
#
#  Run once on each node before the Fedstellar experiment.
#  Adds p2pfl to the existing venv created by setup_nodes.sh.
#
#  Usage:
#    bash experiments/fedstellar/setup_fedstellar.sh [--project-dir <path>]
#
#  Default project dir: auto-detected from script location
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

if [[ ! -f venv/bin/activate ]]; then
    echo "  ERROR: venv not found at $PROJECT_ROOT/venv"
    echo "  Run setup_nodes.sh first to create the virtual environment."
    exit 1
fi

source venv/bin/activate

echo ""
echo "=== Installing p2pfl (Fedstellar) and dependencies ==="
echo "Project dir: $PROJECT_ROOT"
echo ""

pip install "p2pfl[dp]==0.4.4"
pip install "lightning>=2.0.0"
pip install "grpcio>=1.54.0" "grpcio-tools>=1.54.0"
pip install --upgrade datasets huggingface_hub

echo ""
echo "=== Verifying installation ==="
echo ""
python3 -c "
import p2pfl
import lightning
p2pfl_ver = getattr(p2pfl, '__version__', None) or __import__('importlib.metadata', fromlist=['version']).version('p2pfl')
print('p2pfl version    :', p2pfl_ver)
print('lightning version:', lightning.__version__)
print('All OK.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Next step: run the experiment with:"
echo "    bash experiments/fedstellar/run_fedstellar.sh --exp iid --node-id <id>"
echo ""
