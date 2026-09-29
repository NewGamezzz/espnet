#!/bin/bash
# Chained training on 4 A100s in 1 h links: on Delta's gpuA100x4 every 4-GPU
# job that started in a 24 h window had a limit of 1 h or less (132 of them,
# median wait 0 h; none between 4 h and 24 h started; measured 2026-09-08).
# The successor is queued BEFORE srun so a walltime kill continues the chain.
# A link resumes from exp/<tag>/last.ckpt, which espnet3 writes at the end of
# an epoch and nowhere else; epochs are therefore short
# (batch_sampler.batches_per_epoch, about 5 min) and a killed link loses the
# epoch it was in. The chain is bounded by CHAIN_LEFT (default 400 links),
# stops when exp/<tag>/STOP exists, and stops itself after CHAIN_MAX_STALLS
# (default 2) consecutive links that did not advance last.ckpt
# (local/chain_guard.py).
# Usage: sbatch local/submit_train.sbatch [conf/training_....yaml]
#        NO_CHAIN=1 sbatch --time=02:00:00 local/submit_train.sbatch conf/training_smoke.yaml
#SBATCH --job-name=lemas_f5_base
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out
set -euo pipefail
# a crashed rank must not leave multi-GB core files in the recipe dir
ulimit -c 0
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1
# reclaim the allocator's fragmented reserve (5.5 GB seen at OOM on 4 GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONF=${1:-conf/training_f5_base_dualprompt.yaml}
TAG=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['exp_tag'])" "$CONF")
mkdir -p logs
if [ -z "${NO_CHAIN:-}" ]; then
  if ! $PY local/chain_guard.py "exp/$TAG" --max_stalls "${CHAIN_MAX_STALLS:-2}" >> logs/chain.log; then
    echo "chain stopped by local/chain_guard.py, see exp/$TAG/STOP" >&2
    exit 1
  fi
  LEFT=${CHAIN_LEFT:-400}
  if [ "$LEFT" -gt 1 ] && [ ! -f "exp/$TAG/STOP" ]; then
    CHAIN_LEFT=$((LEFT - 1)) sbatch --dependency=afterany:$SLURM_JOB_ID \
      "$SLURM_SUBMIT_DIR/local/submit_train.sbatch" "$CONF" >> logs/chain.log
  else
    echo "chain stopped (CHAIN_LEFT=$LEFT, STOP=$([ -f exp/$TAG/STOP ] && echo yes || echo no))" >> logs/chain.log
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
