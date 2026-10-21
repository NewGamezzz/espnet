#!/bin/bash
# Inspect real training batches before a launch (local/check_dataloader.py).
# Usage: sbatch local/check_dataloader.sbatch [conf/training_....yaml]
#SBATCH --job-name=lemas_loadercheck
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/check_dataloader_%j.out
set -euo pipefail
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export PYTHONUNBUFFERED=1 HF_HOME=${LEMAS_VERSA_CACHE}/huggingface
TRAIN=${1:-conf/training_f5_base_dualprompt_gh200.yaml}
mkdir -p logs
# epochs: the first two and a late one
$PY local/check_dataloader.py --training_config "$TRAIN" \
  --epochs 0 1 1723 --batches 2 --asr_rows 12 \
  --asr_cache "${LEMAS_VERSA_CACHE}/fwhisper_wer" \
  --out "exp/check_dataloader_${SLURM_JOB_ID}"
