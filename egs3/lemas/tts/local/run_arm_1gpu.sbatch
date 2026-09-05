#!/bin/bash
# One evaluation arm (infer + measure) on one GPU within an hour.
# Usage: sbatch local/run_arm_1gpu.sbatch [conf/inference_lemas_eval.yaml] [conf/metrics.yaml]
# Keep at most 3 arms importing at once (Delta Lustre import contention).
#SBATCH --job-name=lemas_arm
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/arm_%j.out
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd ../../.. && pwd)
export PYTHONPATH=$ROOT:$(pwd):${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PATH=/work/nvme/bbjs/ttrachu/espeak-ng/bin:$PATH
export LD_LIBRARY_PATH=/work/nvme/bbjs/ttrachu/espeak-ng/lib:${LD_LIBRARY_PATH:-}
PY=/work/nvme/bbjs/ttrachu/pixi_x86/default/bin/python
INF=${1:-conf/inference_lemas_eval.yaml}
MET=${2:-conf/metrics.yaml}
mkdir -p logs
$PY run.py --stages infer measure --training_config conf/training_f5_base_dualprompt.yaml \
  --inference_config "$INF" --metrics_config "$MET"
