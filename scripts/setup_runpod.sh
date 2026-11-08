#!/usr/bin/env bash
# Set up segway-control-suite for RL training on a remote machine (RunPod, etc.)
#
# Assumes:
#   - Python 3.10+ is active (RunPod PyTorch templates ship 3.10/3.11)
#   - PyTorch is already installed (RunPod template handles CUDA version)
#   - Repo has been cloned to the current directory
#
# Run from the repo root:
#   bash scripts/setup_runpod.sh

set -e

echo "=== Installing segway-control-suite [rl, sim, viz] ==="
pip install -e ".[rl,sim,viz]"

echo ""
echo "=== Smoke test ==="
python - <<'EOF'
from segway.envs.emergency_stop_env import EmergencyStopEnv
env = EmergencyStopEnv()
obs, _ = env.reset(seed=0)
_, r, *_ = env.step([0.0])
print(f"  EmergencyStopEnv OK  obs={obs}  reward={r:.4f}")
EOF

echo ""
echo "=== Done. To train: ==="
echo "  python scripts/train_ebrake_rl.py --steps 500000 --envs 8"
