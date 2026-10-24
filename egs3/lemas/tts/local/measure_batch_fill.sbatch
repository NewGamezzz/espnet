#!/bin/bash
# Real frames per batch against the sampler's bound (local/measure_batch_fill.py)
# for the two layout diagnostics and the two main runs. No audio is read; the
# two 29.6 M-row train manifests take ~2 min and ~28 GB each to load.
# Usage (from the recipe dir on Delta): sbatch local/measure_batch_fill.sbatch
#SBATCH --job-name=lemas_batch_fill
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/batch_fill_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
mkdir -p exp/batch_fill
$PY local/measure_batch_fill.py \
  conf/training_f5_small_en_dualprompt.yaml \
  conf/training_f5_small_en_stock.yaml \
  conf/training_f5_base_dualprompt.yaml \
  conf/training_f5_base_dualprompt_gh200.yaml \
  --epochs 3 --out "exp/batch_fill/batch_fill_$SLURM_JOB_ID.json"
