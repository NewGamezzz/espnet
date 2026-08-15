#!/bin/bash
#SBATCH --job-name=emilia_f5_smoke2
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:v100-32:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke2gpu_%j.out
#
# 2-GPU throughput and memory smoke, run against the 116k-utterance subset
# rather than the full corpus. Produces the number the campaign budget
# depends on (updates/hour at production batch_bins) without waiting for a
# full 8-GPU allocation.
#
# WHY 2 GPUS IS A VALID INSTRUMENT. See the header of
# conf/training_f5_tts_base_smoke_2gpu.yaml for the full argument. In short:
# peak memory is per-rank and espnet3 gives each rank whole batches, so GPU
# count does not affect it; and accumulate_grad_batches is left at the
# production value, so each rank performs the same 4 micro-steps per
# optimizer update it would in production.
#
# NOTE the global batch here is 76,800 frames, a quarter of production. This
# config is an instrument and must never be used for a real training run.
#
# WHY GPU-small. It is a single node (v001) with 8 v100-32. The LibriTTS F5
# Small training normally holds 6 of them, leaving exactly 2 free -- so a
# 2-GPU request starts immediately instead of queueing behind GPU-shared.
# Check first, because if that training is not running the node may be busy
# with something else:
#   sinfo -p GPU-small -o '%.6t %N'
#   scontrol show node v001 | tr ' ' '\n' | grep AllocTRES
#
# `#SBATCH --output=logs/smoke2gpu_%j.out` needs logs/ to exist; Slurm will
# not create it. `mkdir -p logs` before submitting.
#
# ---------------------------------------------------------------------------
# What to read out of the log
# ---------------------------------------------------------------------------
# 1. UPDATES/HOUR: step/time deltas from the MetricsLogger's periodic
#    "Nepoch:train:A-Bbatch: ..." lines (log_every_n_steps: 50). Production
#    at 8 GPUs runs the same 4 micro-steps per rank per update, so this
#    transfers directly, modulo allreduce growing from 2 ranks to 8.
#      wall_clock_hours = 1_200_000 / updates_per_hour
# 2. PEAK GPU MEMORY: the pass condition is simply the absence of a CUDA OOM
#    trace. Expected peak is 24.81 GiB of 31.73 (parity/mem_probe.py under
#    2-rank DDP), so a peak materially above that means the corpus subset's
#    length distribution differs from what the sampler was measured on.
# 3. THE GUARD REMOVAL, END TO END: min_batch_size is 1 while world_size is
#    2, which espnet2's len(batch) >= world_size check used to reject. The
#    "[train] distributed batches: world_size=2, rank=..., total=...,
#    per_rank=..." line appearing at all is the confirmation; a RuntimeError
#    reading "The batch-size must be equal or more than world_size" means the
#    espnet3 tree on this machine is stale and predates commit 342ce1bd06.
# 4. DATALOADER STARVATION: MetricsLogger separates iter_time (which includes
#    dataloader wait) from forward/backward/optim_step. If iter_time is a
#    large fraction, raise dataloader num_workers -- but note 2 tasks x 5 cpus
#    is already 10 of this node's 40 CPUs and the rest belong to whatever
#    else shares v001.
# ---------------------------------------------------------------------------

set -euo pipefail
ROOT=/ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
cd "$ROOT"
# See submit_smoke.sbatch: activate_python.sh runs `eval "$(pixi shell-hook)"`
# and cannot be sourced under `set -u`.
set +eu
source tools/activate_python.sh
set -eu
# Absolute, and re-anchored to $ROOT: the activation leaves the shell in the
# pixi manifest's own tools/ directory, which is a different espnet tree.
cd "$ROOT/egs3/emilia/tts"

# The smoke subset lives outside the recipe tree; the config reads both of
# these from the environment and defaults them to "." when unset, so the
# committed YAML carries no machine-specific paths.
export EMILIA_SMOKE_ROOT="$ROOT/smoke"
export EMILIA_RECIPE_ROOT="$ROOT/egs3/emilia/tts"

# rjieba is absent from the shared pixi env and F5PinyinPreprocessor imports
# it lazily, so without this the failure surfaces mid-training rather than at
# startup. pylibs/ is an isolated copy; the shared env is left alone because
# other queued jobs depend on it.
export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"

# Peak *allocated* is 24.81 GiB of 31.73, but peak *reserved* runs higher as
# variable-length batches fragment the pool -- which is how the LibriTTS run
# died at step 950 (22.60 GiB allocated, 7.45 GiB reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CSVLogger only, no network egress from a compute node.
export WANDB_MODE=offline

echo "python: $(which python)"
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun python -u run.py --stages train \
  --training_config conf/training_f5_tts_base_smoke_2gpu.yaml
echo "SMOKE2GPU_EXIT=$?"
date
echo "SMOKE2GPU_DONE"
