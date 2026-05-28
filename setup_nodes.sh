#!/bin/bash
# =============================================================================
#  setup_nodes.sh  --  One-time environment setup on each node
#
#  Run this script once on every machine that will participate in an experiment.
#  Installs Python dependencies, creates venv, and downloads CIFAR-10.
#
#  Usage:
#    bash setup_nodes.sh [--project-dir <path>]
#
#  Default project dir: ~/fl_project
#  Override: bash setup_nodes.sh --project-dir /data/fl_project
# =============================================================================

set -e

PROJECT_DIR="$HOME/fl_project"

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo ""
echo "=== FL Node Setup ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# ---- 1. System packages ----
echo "[1/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3 python3-venv python3-pip git tmux

# ---- 2. Project directory ----
echo "[2/5] Setting up project directory..."
mkdir -p "$PROJECT_DIR/data"
cd "$PROJECT_DIR"

# ---- 3. Virtual environment ----
echo "[3/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ---- 4. Dependencies ----
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision numpy matplotlib

# ---- 5. Download CIFAR-10 ----
echo "[5/5] Downloading CIFAR-10 dataset..."
python3 -c "
import torchvision, torchvision.transforms as T
t = T.Compose([T.ToTensor()])
torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=t)
torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=t)
print('CIFAR-10 ready.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy your project files to $PROJECT_DIR/"
echo "  2. Fill in config.env (copy from config.env.example)"
echo "  3. Run: chmod +x experiments/run_experiment.sh"
echo "  4. Run an experiment: PYTHONPATH=src python3 src/server.py <my_ip>"
echo ""
