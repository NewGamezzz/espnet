#!/bin/bash
# Delta: measure the per-GPU memory and time of a full training step at the
# worst batch shapes of each candidate batch_bins, in fp32 and bf16
# (local/probe_step_memory.py). One GPU: peak memory is a per-rank quantity.
# Needs the shape files of local/delta/prepare_data.sbatch and
# downloads/vocab.txt (see the README).
#
# The default candidates all divide F5's 307,200 frames per update on 4 GPUs:
# batch_bins / 100 mels x accumulate_grad_batches = 76,800 for accumulate
# 8, 6, 5, 4, 3 and 2.
#
# Usage, from the recipe dir (the account is given at submit time):
#   sbatch --account=<gpu account> local/delta/probe_memory.sbatch
#   BATCH_BINS="960000 1920000" sbatch --account=<gpu account> \
#     local/delta/probe_memory.sbatch conf/<other>.yaml
#SBATCH --job-name=emilia_probe_memory
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=logs/probe_memory_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta/env.sh"
CONF=${1:-conf/training_f5_tts_base.yaml}
BATCH_BINS=${BATCH_BINS:-960000 1280000 1536000 1920000 2560000 3840000}
# reclaim the allocator's fragmented reserve, as the training jobs do
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git -C "$ROOT" log -1 --format='tree: %h %s' || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
date
$PY local/delta/check_imports.py "$ROOT"
# word splitting of BATCH_BINS into separate arguments is intended
# shellcheck disable=SC2086
$PY local/probe_step_memory.py --training_config "$CONF" \
  --batch_bins $BATCH_BINS --precisions 32-true bf16-mixed \
  --out "exp/probe/step_memory_${SLURM_JOB_ID}.json"
date
echo PROBE_MEMORY_DONE
