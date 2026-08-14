#!/bin/bash
#SBATCH --job-name=emilia_f5_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#
# Production Base training, chained across Bridges-2's walltime limit.
# `#SBATCH --output=logs/train_%j.out` requires logs/ to exist -- Slurm does
# not create the parent directory. `mkdir -p logs` before the first submit.
#
# ---------------------------------------------------------------------------
# Quota guard (Task 13 step 1, local/quota_guard.sh):
# ---------------------------------------------------------------------------
# Runs FIRST, before anything is written, so a low-space condition dies the
# job cleanly instead of writing a partial checkpoint into a nearly-full
# shared filesystem the way step96048.ckpt and step96600.ckpt were
# truncated in July 2026.
#
# ---------------------------------------------------------------------------
# Chaining (Task 13 step 2):
# ---------------------------------------------------------------------------
# `fit.ckpt_path: ${exp_dir}/last.ckpt` (already set in
# conf/training_f5_tts_base.yaml) makes each job resume where the last left
# off. This script re-submits itself with `--dependency=afterany` so the
# chain survives the 48h walltime cap.
#
# `--dependency=afterany` fires on success, failure AND cancellation alike
# -- a bad config, a quota-guard abort, or a first-batch OOM would otherwise
# resubmit forever on a shared allocation. Two guards against that:
#   1. CHAIN_DEPTH, exported across the chain via `sbatch --export`, aborts
#      resubmission past MAX_CHAIN_DEPTH (default 500 -- far beyond what
#      1.2M updates at any plausible per-job step count needs; a chain
#      actually reaching this means something is wrong and warrants
#      investigation, not more auto-resubmission).
#   2. A completion check runs BEFORE training: if `last.ckpt` already
#      points at a checkpoint whose step count is >= trainer.max_steps, or
#      an operator has dropped a STOP sentinel file in exp_dir, the job
#      exits 0 (done, not an error) without training or resubmitting.
#
# ---------------------------------------------------------------------------
# Rank-0 OOM diagnostic (Task 13 step 2's third bullet):
# ---------------------------------------------------------------------------
# The plan describes this as "wrap the training step in a try/except that
# logs torch.cuda.memory_summary() and calls dist.abort()". DEVIATION: this
# implementation is NOT a Python change. torch.distributed has no `abort()`
# in the installed torch (2.12.0; verified: `hasattr(torch.distributed,
# "abort")` is False), and the failure this exists to catch --
# [[psc-conversational-f5-oom-hang]]: ranks 1-7 sit in
# lightning_module.py's `_sync2skip` `torch.distributed.barrier()` waiting
# for a dead rank 0 until the 30-min NCCL watchdog times out, which shows up
# as an ALLREDUCE timeout, not an OOM -- is a launcher-level problem, not a
# training-step-level one: rank 0 dies with a normal Python OOM exception
# (PyTorch's own CUDA OOM message already reports allocated/reserved/
# capacity, so wrapping it to call memory_summary() again adds little), and
# the masking happens because Slurm/srun does not by default tear down
# sibling tasks when one exits. So the diagnostic here is:
#   - `srun --kill-on-bad-exit=1`: one dead rank kills the whole step
#     immediately instead of leaving the rest in the barrier.
#   - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`: makes the NCCL watchdog abort
#     promptly on a collective failure instead of hanging (this is the
#     current torch 2.x env var name; it was `NCCL_ASYNC_ERROR_HANDLING`
#     before ~2.2, so re-verify against the installed torch version if this
#     script is ever ported to a different environment).
#   - `NCCL_DEBUG=WARN` for a diagnosable log if it happens again.
# If a training job dies at the first batch with what LOOKS like an
# ALLREDUCE/watchdog timeout, read that as "look for a rank crash first" --
# see README.md's "Rank-0 OOM masquerading as an NCCL timeout" section and
# `collect_stacks.sh` from the same memory note for live triage.
# ---------------------------------------------------------------------------

set -euo pipefail
cd /ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
source tools/activate_python.sh
cd egs3/emilia/tts

bash local/quota_guard.sh

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN_DEPTH="${MAX_CHAIN_DEPTH:-500}"
if (( CHAIN_DEPTH >= MAX_CHAIN_DEPTH )); then
    echo "ABORT: chain depth $CHAIN_DEPTH reached MAX_CHAIN_DEPTH=$MAX_CHAIN_DEPTH;" \
        "not resubmitting. Investigate before raising the cap." >&2
    exit 1
fi

CONFIG="${CONFIG:-conf/training_f5_tts_base.yaml}"
read -r EXP_DIR MAX_STEPS <<< "$(python -c "
from pathlib import Path
from espnet3.utils.config_utils import load_and_merge_config
cfg = load_and_merge_config(
    Path('$CONFIG'), config_name='training_f5_tts_base.yaml'
)
print(cfg.exp_dir, cfg.trainer.max_steps)
")"

if [[ -f "$EXP_DIR/STOP" ]]; then
    echo "STOP sentinel found at $EXP_DIR/STOP; not training, not resubmitting."
    exit 0
fi

LAST_CKPT="$EXP_DIR/last.ckpt"
if [[ -e "$LAST_CKPT" ]]; then
    # save_last="link" (default_callbacks.py) makes last.ckpt a symlink to
    # step{step}.ckpt; resolve the target and read the step count back out
    # of the filename rather than loading the checkpoint.
    TARGET_NAME="$(basename "$(readlink -f "$LAST_CKPT" 2>/dev/null || echo "$LAST_CKPT")")"
    STEP="$(echo "$TARGET_NAME" | grep -oE '[0-9]+' | head -1 || true)"
    if [[ -n "$STEP" ]] && (( STEP >= MAX_STEPS )); then
        echo "$LAST_CKPT is at step $STEP >= max_steps $MAX_STEPS; training is" \
            "complete, not resubmitting."
        exit 0
    fi
fi

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

srun --kill-on-bad-exit=1 python run.py --stages train \
    --training_config "$CONFIG"

# Chain across the walltime limit. --export=ALL forwards CHAIN_DEPTH (and
# everything else already in this job's environment) to the next job;
# without it, CHAIN_DEPTH would silently reset to 0 on every hop and the
# cap above would never bind.
sbatch --export=ALL,CHAIN_DEPTH=$((CHAIN_DEPTH + 1)) \
    --dependency=afterany:"$SLURM_JOB_ID" \
    local/submit_train.sbatch
