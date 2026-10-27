#!/bin/bash
# Loss of a checkpoint on every validation batch and on training batches
# (local/loss_by_split.py): is the valid/train loss gap the length-sorted
# validation sampler or the model? One GPU, under an hour.
# Usage: sbatch local/loss_by_split.sbatch conf/training_....yaml exp/<tag>/last.ckpt
#SBATCH --job-name=lemas_loss_by_split
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:50:00
#SBATCH --output=logs/loss_by_split_%j.out
set -euo pipefail
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export PYTHONUNBUFFERED=1
TRAIN=$1
CKPT=$2
mkdir -p logs
$PY local/loss_by_split.py --training_config "$TRAIN" --ckpt "$CKPT" \
  --out "exp/loss_by_split/${SLURM_JOB_ID}.json"
