#!/bin/bash
# Body of one training chain link, sourced by local/submit_train*.sbatch after
# the cluster's env script. The sbatch file sets SELF (its own file name, the
# successor it queues), CHAIN_SUFFIX (log name suffix: Delta and DeltaAI job
# ids collide) and CONF_DEFAULT; the first argument overrides the config.
#
# The successor is queued BEFORE srun so a walltime kill continues the chain.
# A link resumes from exp/<tag>/last.ckpt, which espnet3 writes at the end of
# an epoch and nowhere else; epochs are therefore short
# (batch_sampler.batches_per_epoch, 100 steps) and a killed link loses the
# epoch it was in. The chain is bounded by CHAIN_LEFT (default 400 links),
# stops when exp/<tag>/STOP exists, and stops itself after CHAIN_MAX_STALLS
# (default 2) consecutive links that did not advance last.ckpt
# (local/chain_guard.py). NO_CHAIN=1 runs one link and queues nothing.
set -euo pipefail
# a crashed rank must not leave multi-GB core files in the recipe dir
ulimit -c 0
# $0 is Slurm's spool copy of the sbatch file; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
# reclaim the allocator's fragmented reserve (5.5 GB seen at OOM on 4 GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONF=${1:-$CONF_DEFAULT}
TAG=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['exp_tag'])" "$CONF")
CHAIN_LOG=logs/chain${CHAIN_SUFFIX}.log
mkdir -p logs
if [ -z "${NO_CHAIN:-}" ]; then
  if ! $PY local/chain_guard.py "exp/$TAG" --max_stalls "${CHAIN_MAX_STALLS:-2}" >> "$CHAIN_LOG"; then
    echo "chain stopped by local/chain_guard.py, see exp/$TAG/STOP" >&2
    exit 1
  fi
  LEFT=${CHAIN_LEFT:-400}
  if [ "$LEFT" -gt 1 ] && [ ! -f "exp/$TAG/STOP" ]; then
    CHAIN_LEFT=$((LEFT - 1)) sbatch --dependency=afterany:$SLURM_JOB_ID \
      "$SLURM_SUBMIT_DIR/local/$SELF" "$CONF" >> "$CHAIN_LOG"
  else
    echo "chain stopped (CHAIN_LEFT=$LEFT, STOP=$([ -f exp/$TAG/STOP ] && echo yes || echo no))" >> "$CHAIN_LOG"
  fi
fi
CKPT=exp/$TAG/last.ckpt
if [ -L "$CKPT" ] && [ ! -e "$CKPT" ]; then
  echo "$CKPT points to a missing file; refusing to restart from scratch" >&2
  exit 1
fi
EXTRA=""
if [ -f "$CKPT" ]; then EXTRA="--ckpt_path $CKPT"; fi
# one dead rank ends the job: the others would idle in NCCL until the walltime
srun --kill-on-bad-exit=1 $PY run.py --stages train --training_config "$CONF" $EXTRA
