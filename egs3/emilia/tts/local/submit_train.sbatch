#!/bin/bash
#SBATCH --job-name=emilia_f5_train
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#
# Production Base training, chained across Bridges-2's walltime limit.
# `#SBATCH --output=logs/train_%j.out` requires logs/ to exist -- Slurm does
# not create the parent directory. `mkdir -p logs` before the first submit.
#
# ---------------------------------------------------------------------------
# Quota guard (Task 13 step 1, local/quota_guard.sh):
# ---------------------------------------------------------------------------
# Runs before anything is written, so a low-space condition dies the job
# cleanly instead of writing a partial checkpoint into a nearly-full shared
# filesystem the way step96048.ckpt and step96600.ckpt were truncated in
# July 2026. This deliberately halts the chain (see below) rather than
# resubmitting through a sustained low-space condition -- the design intent
# is a watchdog that stops submission until an operator reclaims space, not
# an infinite retry loop.
#
# ---------------------------------------------------------------------------
# Chaining (Task 13 step 2) -- FIX ROUND 1: resubmission moved to BEFORE
# training, not after it
# ---------------------------------------------------------------------------
# This script computes $LAST_CKPT and, when it already exists on disk,
# passes `--ckpt_path "$LAST_CKPT"` to `run.py` (which sets
# `fit.ckpt_path` for that invocation only -- see step 7 below and
# conf/training_f5_tts_base.yaml's `fit: {}`) so each job resumes where the
# last left off. This script re-submits itself with `--dependency=afterany`
# so the chain survives the 48h walltime cap.
#
# CORRECTED PLACEMENT: the next hop is queued BEFORE `srun` runs, not after
# it. The original placement (queue after `srun` returns) is broken: when
# Bridges-2 hits the --time limit, Slurm delivers SIGTERM to the job's
# entire process tree, including this parent bash script, not just the srun
# step. Bash's default disposition for SIGTERM in a non-interactive script
# with no trap is immediate termination, so a trailing resubmit line never
# runs on a walltime-limited job -- exactly the primary scenario this
# chaining exists for. The chain would silently stop at every walltime
# boundary with no error, noticeable only because the queue went empty.
# Queuing before training instead means the successor is already in the
# queue regardless of how THIS job later dies: clean exit, walltime SIGTERM,
# node failure, preemption, or `scancel`.
#
# `--dependency=afterany` fires on success, failure AND cancellation alike
# -- an unguarded chain would resubmit forever on a shared allocation if a
# job died instantly (bad config, a quota-guard abort, a first-batch OOM).
# Since the next hop is now queued unconditionally (before training even
# starts), runaway resubmission is prevented at the TOP of the *next* job
# instead of by conditional submission at the bottom of this one:
#   1. CHAIN_DEPTH, exported across the chain via `sbatch --export`, aborts
#      -- before doing anything else, including queuing a further hop --
#      once resubmission has happened MAX_CHAIN_DEPTH times (default 500 --
#      far beyond what 1.2M updates at any plausible per-job step count
#      needs; a chain actually reaching this means something is wrong and
#      warrants investigation, not more auto-resubmission).
#   2. A completion check: if `last.ckpt` already points at a checkpoint
#      whose step count is >= trainer.max_steps, or an operator has dropped
#      a STOP sentinel file in exp_dir, the job exits 0 (done, not an
#      error) WITHOUT queuing a further hop. This check runs, and must run,
#      BEFORE the next-hop queuing step below -- otherwise a finished run
#      would queue one more job that then immediately exits, forever.
#
# SCANCEL BEHAVIOUR (operator-facing, since --dependency=afterany fires on
# cancellation too): once a job has passed the queuing step below, its
# successor is already sitting in the queue. `scancel`-ing the currently
# running job does NOT cancel that already-queued successor -- this is a
# direct consequence of using `afterany` (needed so the chain also survives
# preemption, which looks like a cancellation to Slurm). To actually stop
# the chain:
#   - Preferred: touch `${exp_dir}/STOP`. The already-queued successor will
#     start, hit the completion check above BEFORE training or queuing
#     anything further, and exit 0. A brief, harmless job start is the
#     accepted cost of guaranteed resume-after-preemption elsewhere in the
#     chain.
#   - Alternative: also `scancel` the queued successor's job id (from
#     `squeue -u $USER`) if even that brief start is undesirable.
# This is also documented in README.md's chaining section.
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

# 8h walltime, because that is GPU-small's ceiling: the gpusmallpartition QOS
# caps MaxWall at 08:00:00 and a longer --time is rejected outright at
# submission. GPU-shared would allow 48h; the trade was decided in favour of
# GPU-small's dedicated node.
#
# That makes this a ~300-segment chain for 1.2M updates, so MAX_CHAIN_DEPTH is
# 800 rather than 500: at the measured floor of ~324 updates/hour an 8h segment
# is ~2,600 updates, which needs ~460 hops, and a cap of 500 could stop the run
# just short of the target. 800 still bounds a runaway.
#
# Per-segment overhead is small: sampler construction on the real 37.5M-row
# shape file measured 207.9 s (job 43823280), i.e. 0.7% of an 8h segment.
#
# 4 x v100-32 on GPU-small, matching conf/training_f5_tts_base.yaml's
# num_device: 4 and accumulate_grad_batches: 8, which with batch_bins 960000
# gives 4 * 8 * 960000 / 100 = 307,200 mel-frames per optimizer update -- the
# paper's Sec. 4 figure for the Base model behind Tables 1 and 2, verified
# against the paper text rather than inherited from a comment.
#
# GPU-small is a SINGLE node (v001) with 8 v100-32 shared against the LibriTTS
# F5 Small training. A 4-GPU request needs 4 of them free; check before
# submitting, or the job simply queues:
#   scontrol show node v001 | tr ' ' '\n' | grep AllocTRES
#
# ntasks-per-node MUST equal the GPU count: Lightning DDP spawns one process
# per rank via srun, and a mismatch either strands GPUs or oversubscribes them.

set -euo pipefail
ROOT=/ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
cd "$ROOT"
# `set -e`/`set -u` are disabled across the activation ONLY. activate_python.sh
# runs `eval "$(pixi shell-hook)"`, whose output references unset variables, so
# under `set -u` the job aborts before producing any output: observed on PSC as
# job 43566289, FAILED in 3 seconds with a 0-byte log. Re-enabled immediately
# afterwards so the rest of the script keeps strict-mode semantics.
set +eu
source tools/activate_python.sh
set -eu
# Absolute, and re-anchored to $ROOT. activate_python.sh runs
# `eval "$(pixi shell-hook)"`, which leaves the shell in the pixi manifest's
# own tools/ directory -- and because tools/ is shared across the espnet
# trees on this allocation, that is a DIFFERENT tree
# (espnet_f5_libritts/tools), not this one. A relative `cd egs3/emilia/tts`
# after the activation therefore resolves against the wrong root and the job
# dies before it starts: observed as job 43630646, which failed in seconds
# with `cd: egs3/emilia/tts: No such file or directory`.
cd "$ROOT/egs3/emilia/tts"

# MUST be set, and set BEFORE the config-reading python block below, which
# already imports espnet3. Without it, `espnet3` and `egs3` resolve out of the
# pixi env's installed espnet -- which lives in the espnet_f5_libritts tree --
# so training imported espnet3 from THAT tree while loading src/ and dataset/
# from this one, and died with `ModuleNotFoundError: No module named
# 'egs3.emilia'` on every rank (job 43825659). It also silently bypassed the
# espnet3 fixes this branch carries, and pylibs/ (rjieba, absent from the
# shared env and needed lazily by F5PinyinPreprocessor).
#
# The pre-flight check below does NOT catch this: it tests that artifacts
# exist on disk, not that the code is importable.
export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"

# Guards the failure mode that is NOT peak allocation: the allocator. Peak
# allocated at the configured batch_bins is 24.81 GiB of 31.73 under 2-rank
# DDP (parity/mem_probe.py), but peak *reserved* runs higher as
# variable-length batches fragment the pool. The LibriTTS F5 Small run failed
# exactly there -- it survived 950 steps, then OOM'd with only 22.60 GiB
# allocated against 7.45 GiB reserved-but-unallocated. expandable_segments
# lets the allocator grow a segment instead of stranding it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single node, so there is no inter-node traffic and InfiniBand has no
# legitimate role -- but NCCL will still choose it for GPU pairs that lack a
# P2P path, and that is what hung job 43826521.
#
# GPU-small is shared, so Slurm can hand out a 4-GPU set that straddles both
# PCIe root complexes. Diagnostic 43832550 caught exactly that:
#     rank 0 busId 16000   rank 1 busId 3a000
#     rank 2 busId 3b000   rank 3 busId 89000
#     0<->1, 1<->2  via P2P/CUMEM
#     0<->3, 2<->3  via NET/IB/*/GDRDMA      <-- over the fabric, same chassis
# That link completed 455 collectives and then stalled forever, which is why
# the failure looked like a mid-training deadlock rather than a startup error.
#
# Disabling IB forces NCCL to fall back to shared memory for the non-P2P
# pairs. SHM is slower than NVLink but is reliable and stays on-node, and for
# a single-node job it costs nothing that IB was providing.
export NCCL_IB_DISABLE=1

# 1) Chain-depth cap, checked before anything else (including the quota
#    guard): if this job's own depth already reached the cap, stop here,
#    unconditionally, without training or queuing a further hop.
# 0) Duplicate-chain guard. If another job of this name is already RUNNING,
#    this one belongs to a forked chain: exit WITHOUT training and WITHOUT
#    queuing a successor, so the surplus chain terminates here instead of
#    self-perpetuating. Two trainers sharing exp_dir corrupt each other's
#    checkpoints -- they race on last.ckpt and Lightning silently writes
#    "-v1" variants rather than failing.
#
#    This cannot misfire on the legitimate handoff: the successor is queued
#    with --dependency=afterany on its parent, so the parent has already
#    terminated before the successor starts and cannot be seen RUNNING.
SIBLINGS=$(squeue -u "$USER" -n emilia_f5_train -t RUNNING -h -o '%i' 2>/dev/null \
    | grep -v "^${SLURM_JOB_ID}$" | wc -l)
if (( SIBLINGS > 0 )); then
    echo "ABORT: $SIBLINGS other emilia_f5_train job(s) already RUNNING:"
    squeue -u "$USER" -n emilia_f5_train -t RUNNING -o '%.10i %.11M %.14R' || true
    echo "This is a duplicate chain. Exiting without training and without" \
        "resubmitting, so it dies here rather than forking further."
    exit 0
fi

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN_DEPTH="${MAX_CHAIN_DEPTH:-800}"
if (( CHAIN_DEPTH >= MAX_CHAIN_DEPTH )); then
    echo "ABORT: chain depth $CHAIN_DEPTH reached MAX_CHAIN_DEPTH=$MAX_CHAIN_DEPTH;" \
        "not resubmitting. Investigate before raising the cap." >&2
    exit 1
fi

# 2) Quota guard: deliberately halts the chain (see header) rather than
#    queuing through a sustained low-space condition.
bash local/quota_guard.sh

# 3) Load config. $CONFIG is passed via an environment variable, not
#    interpolated into the python source string, so a path containing a
#    single quote (or any other shell-meaningful character) cannot break
#    the inline script.
# NOTE: `read ... <<< "$(cmd)"` would NOT fail under `set -e` if `cmd`
# raised -- bash only checks the exit status of `read` itself, not the
# substitution feeding it, so a broken config would silently leave these
# variables empty instead of aborting. Capture into CONFIG_INFO first: a
# bare `var=$(cmd)` IS a checked simple command under `set -e`. Only THEN
# is CONFIG_INFO (now a plain string, not a live substitution) fed to
# `mapfile`, so a python failure aborts here rather than downstream with
# blank variables. One value per line, not space-joined on one line: this
# now carries seven fields (Task 13's original two, plus the pre-flight
# artifact paths below), and `read -r a b <<< "..."` silently mis-splits
# once a value can itself contain whitespace-adjacent tokens.
export CONFIG="${CONFIG:-conf/training_f5_tts_base.yaml}"
CONFIG_INFO="$(python -c "
import os
from pathlib import Path
from espnet3.utils.config_utils import load_and_merge_config
cfg = load_and_merge_config(
    Path(os.environ['CONFIG']), config_name='training.yaml'
)
print(cfg.exp_dir)
print(cfg.trainer.max_steps)
print(cfg.vocab_file)
print(cfg.create_shape.manifest_paths.train)
print(cfg.create_shape.manifest_paths.valid)
print(cfg.dataloader.train.iter_factory.batches.shape_files[0])
print(cfg.dataloader.valid.iter_factory.batches.shape_files[0])
")"
mapfile -t CONFIG_LINES <<< "$CONFIG_INFO"
EXP_DIR="${CONFIG_LINES[0]}"
MAX_STEPS="${CONFIG_LINES[1]}"
VOCAB_FILE="${CONFIG_LINES[2]}"
TRAIN_MANIFEST="${CONFIG_LINES[3]}"
VALID_MANIFEST="${CONFIG_LINES[4]}"
TRAIN_SHAPE="${CONFIG_LINES[5]}"
VALID_SHAPE="${CONFIG_LINES[6]}"

# 4) Completion check, BEFORE queuing the next hop: a finished run must not
#    queue one more job that then immediately exits, forever.
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

# 5) Pre-flight artifact check, BEFORE queuing the next hop (IMPORTANT 1):
#    `d54c4d225d` moved successor-queuing to before `srun`, which means a
#    fast first-batch failure (e.g. a missing vocab_file or feats_shape --
#    exactly CRITICAL 3's failure mode) now queues its own successor before
#    ever finding out training can't start, and that successor repeats the
#    same failure, and so on -- up to MAX_CHAIN_DEPTH=500 queued-and-failing
#    jobs from one bad prerequisite. A handful of existence checks here
#    (vocab_file, both feats_shape files, both manifests) turn that into
#    one failed job instead.
MISSING=0
for ARTIFACT in "$VOCAB_FILE" "$TRAIN_SHAPE" "$VALID_SHAPE" \
    "$TRAIN_MANIFEST" "$VALID_MANIFEST"; do
    if [[ ! -e "$ARTIFACT" ]]; then
        echo "PRE-FLIGHT FAILED: required artifact missing: $ARTIFACT" >&2
        MISSING=1
    fi
done
if (( MISSING )); then
    echo "ABORT: not training, not resubmitting -- run create_dataset and" \
        "create_shape first." >&2
    exit 1
fi

# 6) Queue the next hop NOW, before training runs -- see the header comment
#    for why this must happen before `srun`, not after it. --export=ALL
#    forwards CHAIN_DEPTH (and everything else already in this job's
#    environment) to the next job; without it, CHAIN_DEPTH would silently
#    reset to 0 on every hop and the cap in step 1 would never bind.
# `singleton` is not optional. --dependency=afterany fires on CANCELLATION as
# well as success/failure (documented above), so every `scancel` of a running
# job releases its already-queued successor, which then queues its own -- a
# cancelled chain does not die, it forks. Combined with manual resubmissions,
# that produced 90 jobs and, on 2026-08-21, TWO training jobs writing the same
# exp_dir concurrently: Lightning wrote step8000-v1.ckpt and step9200-v1.ckpt
# because the un-suffixed names already existed, and last.ckpt ended up
# pointing at the LAGGING job's step while another was 50 steps ahead.
#
# `singleton` tells Slurm to hold this job until no other job of the same name
# and user is running, so duplicate chains serialise instead of racing. The
# guard in step 0 then prunes them down to one.
sbatch --export=ALL,CHAIN_DEPTH=$((CHAIN_DEPTH + 1)) \
    --dependency=afterany:"$SLURM_JOB_ID",singleton \
    local/submit_train.sbatch

# 7) Train. If this dies for any reason -- clean completion short of
#    max_steps, walltime SIGTERM, node failure, preemption, manual scancel
#    -- the successor queued in step 6 is already there to pick it up.
#    conf/training_f5_tts_base.yaml ships `fit: {}` deliberately (an
#    unconditional fit.ckpt_path FileNotFoundErrors Lightning on a fresh
#    run, verified against the installed lightning==2.6.5): --ckpt_path is
#    passed here, to run.py, only when $LAST_CKPT already exists on disk.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

CKPT_ARGS=()
if [[ -e "$LAST_CKPT" ]]; then
    CKPT_ARGS=(--ckpt_path "$LAST_CKPT")
fi

srun --kill-on-bad-exit=1 python run.py --stages train \
    --training_config "$CONFIG" "${CKPT_ARGS[@]}"
