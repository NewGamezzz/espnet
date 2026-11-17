#!/bin/bash
# Delta: F5-TTS Base training on one gpuA100x4 node, as a chain of jobs
# (conf/training_f5_tts_base_delta.yaml). Needs the manifests and shape files
# of local/delta/prepare_data.sbatch and downloads/vocab.txt (see the README).
#
# One job is one link. A link
#   - stops at once, without training, when exp/<exp_tag>/STOP exists;
#   - asks local/chain_guard.py whether the links before it advanced
#     last.ckpt, and stops the chain (it writes STOP) after CHAIN_MAX_STALLS
#     (default 2) links in a row that did not;
#   - queues its successor BEFORE it trains, because the walltime kill ends
#     this script too. The successor waits for this job (afterany), is
#     charged to the same account, and counts CHAIN_LEFT down (default 40
#     links);
#   - resumes from exp/<exp_tag>/last.ckpt when it exists. espnet3 writes that
#     file at the end of an epoch and nowhere else, and the config keeps
#     epochs short, so a killed link loses only the epoch it was in.
# To stop a chain: touch exp/<exp_tag>/STOP (the queued successor then ends
# in seconds), or scancel the successor as well as the running link.
#
# Why links are long: a 4-GPU job waits a day or more in this queue, and a
# job held by a dependency gains no priority while it waits.
#
# What to check in logs/train_<jobid>.out of a resumed link: "Restored all
# states", a first step one above the checkpoint's, and a learning rate that
# continues the schedule. logs/chain_<exp_tag>.log has one entry per link.
#
# Usage, from the recipe dir (the account is given at submit time):
#   sbatch --account=<gpu account> local/delta/train.sbatch
# One link of another config, queueing nothing (the trial config's header
# shows the two-job check of the resume path on the interactive partition):
#   NO_CHAIN=1 sbatch --account=<gpu account> --partition=gpuA100x4-interactive \
#     --time=00:30:00 local/delta/train.sbatch conf/training_f5_tts_base_delta_trial.yaml
#SBATCH --job-name=emilia_f5_base
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
set -euo pipefail
# a crashed rank must not leave multi-GB core files in the recipe dir
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta/env.sh"
CONF=${1:-conf/training_f5_tts_base_delta.yaml}
# reclaim the allocator's fragmented reserve, which variable-length batches grow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
git -C "$ROOT" log -1 --format='tree: %h %s' || true
echo "config: $CONF  corpus: $EMILIA_CORPUS_ROOT ($EMILIA_CORPUS_LAYOUT)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
date
$PY local/delta/check_imports.py "$ROOT"
TAG=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['exp_tag'])" "$CONF")
EXP=exp/$TAG
mkdir -p logs
CHAIN_LOG=logs/chain_$TAG.log

if [ -f "$EXP/STOP" ]; then
  echo "$EXP/STOP exists: not training, not queueing a successor"
  exit 0
fi

if [ -z "${NO_CHAIN:-}" ]; then
  echo "link $SLURM_JOB_ID $(date '+%F %T')" >> "$CHAIN_LOG"
  if ! $PY local/chain_guard.py "$EXP" --max_stalls "${CHAIN_MAX_STALLS:-2}" >> "$CHAIN_LOG"; then
    echo "chain stopped by local/chain_guard.py, see $EXP/STOP" >&2
    exit 1
  fi
  LEFT=${CHAIN_LEFT:-40}
  if [ "$LEFT" -gt 1 ]; then
    # A link that cannot queue its successor still trains: its checkpoint is
    # what the next submission continues from.
    CHAIN_LEFT=$((LEFT - 1)) sbatch --account="$SLURM_JOB_ACCOUNT" \
      --dependency=afterany:"$SLURM_JOB_ID" \
      "$SLURM_SUBMIT_DIR/local/delta/train.sbatch" "$CONF" >> "$CHAIN_LOG" 2>&1 ||
      echo "WARNING: the successor was not queued (see $CHAIN_LOG); the chain ends with this link" >&2
  else
    echo "chain ends with this link (CHAIN_LEFT=$LEFT)" >> "$CHAIN_LOG"
  fi
fi

CKPT=$EXP/last.ckpt
# a kill in the middle of a save can leave last.ckpt pointing at a deleted file
$PY local/chain_guard.py "$EXP" --repair_last
if [ -L "$CKPT" ] && [ ! -e "$CKPT" ]; then
  echo "$CKPT points to a missing file; refusing to restart from scratch" >&2
  exit 1
fi
EXTRA=""
if [ -f "$CKPT" ]; then
  EXTRA="--ckpt_path $CKPT"
  echo "resuming from $CKPT -> $(readlink "$CKPT" || true)"
else
  echo "no $CKPT: starting from scratch"
fi

# memory.used per GPU once a minute; batches are shuffled, so the peak of a
# link is "no out-of-memory so far", not the worst case
GPU_LOG=logs/train_gpu_${SLURM_JOB_ID}.csv
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 60 > "$GPU_LOG" 2>/dev/null &
SMI_PID=$!
rc=0
# one dead rank ends the job: the others would idle in NCCL until the walltime
srun --kill-on-bad-exit=1 $PY run.py --stages train --training_config "$CONF" $EXTRA || rc=$?
kill "$SMI_PID" 2>/dev/null || true
date
echo "TRAIN_LINK_DONE rc=$rc"
exit "$rc"
