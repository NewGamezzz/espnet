#!/bin/bash
#SBATCH --job-name=emilia_ab_revert
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --time=01:30:00
#SBATCH --output=logs/ab_revert_%j.out
#
# A/B: run training with the PRISTINE base-branch dataloader.py, to test
# whether this branch's only non-recipe source change causes the NCCL hang in
# job 43826521.
#
# The three reverted changes:
#   1. build_iter(self.epoch, shuffle=False)  <- restored (was: shuffle honoured)
#   2. len(batch) < world_size guard          <- restored
#   3. mode threading                          <- reverted (logging only)
#
# READ THE RESULT CAREFULLY. #1 changes BATCH ORDER, not batch count:
# sort_batch is ascending, so with shuffle=False the epoch runs SHORT batches
# first and the 12,000-frame peak arrives only near the end. With shuffle=True
# a peak batch can land at step 1. Rank 0 measured 27.8 GiB of 31.73 (87.6%)
# during the failure, so the leading hypothesis is that an early long batch
# OOMs rank 0, the skip path desyncs ranks through _sync2skip's barrier
# (lightning_module.py:207), and NCCL deadlocks -- which is a documented
# failure mode in this project.
#
# If that is what is happening, THIS RUN WILL PASS FOR THE WRONG REASON: it
# will simply not reach a long batch inside 90 minutes. A pass here therefore
# means "shuffle order is implicated", NOT "the base dataloader is correct and
# we are done". The follow-up that distinguishes them is keeping shuffle on and
# lowering peak memory (batch_bins 960000 -> 480000 with accum 8 -> 16, which
# holds the 307,200-frame global batch and halves the peak batch).
#
# Not chained: a systematic failure must not spawn successors.

set -euo pipefail
ROOT=/ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
DL="$ROOT/espnet3/components/data/dataloader.py"

# Always put the branch's own version back, however this job ends.
restore() {
    cp -f "$ROOT/diag/dataloader_FIXED.py" "$DL"
    echo "RESTORED branch dataloader.py"
}
trap restore EXIT

cp -f "$ROOT/diag/dataloader_BASE.py" "$DL"
echo "SWAPPED IN base dataloader.py:"
grep -n "build_iter(self.epoch" "$DL" || true

cd "$ROOT"
set +eu
source tools/activate_python.sh
set -eu
cd "$ROOT/egs3/emilia/tts"

export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TORCH_NCCL_TRACE_BUFFER_SIZE=8192
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$ROOT/diag/ab_${SLURM_JOB_ID}_trace_rank"

echo "python: $(which python)"; date
srun python -u run.py --stages train \
  --training_config conf/training_f5_tts_base.yaml || true
date
echo AB_REVERT_DONE
