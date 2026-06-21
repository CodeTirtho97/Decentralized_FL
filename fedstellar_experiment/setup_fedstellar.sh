#!/bin/bash
# =============================================================================
#  setup_fedstellar.sh  --  Install p2pfl (Fedstellar) and dependencies
#
#  Run once on each EC2 node before the Fedstellar experiment.
#  Adds p2pfl to the existing venv created by the main experiment setup.
#
#  Usage:
#    bash fedstellar_experiment/setup_fedstellar.sh
# =============================================================================

set -e

cd ~/fl_project
source venv/bin/activate

echo ""
echo "=== Installing p2pfl (Fedstellar) and dependencies ==="
echo ""

pip install "p2pfl[dp]==0.4.4"
pip install "lightning>=2.0.0"
pip install "grpcio>=1.54.0" "grpcio-tools>=1.54.0"
pip install --upgrade datasets huggingface_hub   # fix HfFolder removal in newer huggingface_hub

echo ""
echo "=== Verifying installation ==="
echo ""
python3 -c "
import p2pfl
import lightning
p2pfl_ver = getattr(p2pfl, '__version__', None) or __import__('importlib.metadata', fromlist=['version']).version('p2pfl')
print('p2pfl version  :', p2pfl_ver)
print('lightning version:', lightning.__version__)
print('All OK.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Next step: run the experiment with ./launch_fedstellar.sh iid"
echo ""
