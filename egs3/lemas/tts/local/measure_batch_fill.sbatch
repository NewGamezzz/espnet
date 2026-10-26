#!/bin/bash
# Real frames per batch against the sampler's bound (local/measure_batch_fill.py)
# for the two layout diagnostics, the two main runs and the three baseline rows
# (S0c, S0a, D1; their accumulate_grad_batches comes from this measurement; the
# base config is the regression reference at 3,261.45). No audio is read; a
# 29.6 M-row train manifest takes ~2 min and ~28 GB to load.
# Usage (from the recipe dir on Delta): sbatch local/measure_batch_fill.sbatch
#        sbatch local/measure_batch_fill.sbatch conf/<one or more configs>.yaml
# (configs on the command line replace the list below, to re-measure one row)
#SBATCH --job-name=lemas_batch_fill
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/batch_fill_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
mkdir -p exp/batch_fill
CONFS=(
  conf/training_f5_small_en_dualprompt.yaml
  conf/training_f5_small_en_stock.yaml
  conf/training_f5_base_dualprompt.yaml
  conf/training_f5_base_dualprompt_gh200.yaml
  conf/training_f5_base_s0c_stock_phones.yaml
  conf/training_f5_base_s0a_stock_chars.yaml
  conf/training_f5_base_d1_spkonly.yaml
)
if [ "$#" -gt 0 ]; then CONFS=("$@"); fi
$PY local/measure_batch_fill.py "${CONFS[@]}" \
  --epochs 3 --out "exp/batch_fill/batch_fill_$SLURM_JOB_ID.json"
