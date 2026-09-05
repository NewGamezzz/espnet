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
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export PYTHONUNBUFFERED=1
INF=${1:-conf/inference_lemas_eval.yaml}
MET=${2:-conf/metrics.yaml}
mkdir -p logs
$PY run.py --stages infer measure --training_config conf/training_f5_base_dualprompt.yaml \
  --inference_config "$INF" --metrics_config "$MET"
