#!/bin/bash
# Delta: one short trial of the production training step on 4 A100s
# (conf/training_f5_tts_base_delta_trial.yaml), on the interactive partition.
# It starts from scratch in its own exp dir, resumes nothing and queues no
# successor. Needs the manifests and shape files of
# local/delta/prepare_data.sbatch and downloads/vocab.txt (see the README).
#
# What to read out of logs/train_trial_<jobid>.out:
#   - start-up: the time from the first `date` to the first training log line
#     (four ranks each load the manifest and build the batch sampler)
#   - the periodic "Nepoch:train:A-Bbatch: ..." lines: iter_time (which
#     includes waiting for the data loader) against train_time, and from the
#     step/time deltas the updates per hour
#   - one validation pass and "TRAIN_TRIAL_DONE rc=0"
#   - TRIAL_GPU_PEAK_MIB: the highest memory.used seen per GPU, sampled every
#     15 s. Batches are shuffled, so this is "no out-of-memory in N updates",
#     not the worst case (local/delta/probe_memory.sbatch measures that).
# The trial checkpoints (about 5.4 GB each) are not needed afterwards.
#
# Usage, from the recipe dir (the account is given at submit time):
#   sbatch --account=<gpu account> local/delta/train_trial.sbatch
#SBATCH --job-name=emilia_f5_trial
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_trial_%j.out
set -euo pipefail
# a crashed rank must not leave multi-GB core files in the recipe dir
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta/env.sh"
CONF=${1:-conf/training_f5_tts_base_delta_trial.yaml}
# reclaim the allocator's fragmented reserve, which variable-length batches grow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# no run is created on a logging service from a trial
export WANDB_MODE=offline
git -C "$ROOT" log -1 --format='tree: %h %s' || true
echo "config: $CONF  corpus: $EMILIA_CORPUS_ROOT ($EMILIA_CORPUS_LAYOUT)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
date
$PY local/delta/check_imports.py "$ROOT"
GPU_LOG=logs/train_trial_gpu_${SLURM_JOB_ID}.csv
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 15 > "$GPU_LOG" &
SMI_PID=$!
rc=0
# one dead rank ends the job: the others would idle in NCCL until the walltime
srun --kill-on-bad-exit=1 $PY run.py --stages train --training_config "$CONF" || rc=$?
kill "$SMI_PID" 2>/dev/null || true
date
echo "TRIAL_GPU_PEAK_MIB $(awk -F', ' '{ if ($3 > m[$2]) m[$2] = $3 } END { for (i in m) printf "gpu%s=%s ", i, m[i] }' "$GPU_LOG")"
echo "TRAIN_TRIAL_DONE rc=$rc"
exit "$rc"
