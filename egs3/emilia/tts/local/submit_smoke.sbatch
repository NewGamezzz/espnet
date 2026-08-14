#!/bin/bash
#SBATCH --job-name=emilia_f5_smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#
# GPU smoke: produces the number the whole campaign budget depends on.
# Trains conf/training_f5_tts_base_smoke.yaml, which is the production Base
# config (training_f5_tts_base.yaml) with ONLY max_steps, val_check_interval,
# limit_val_batches, exp_tag and trainer.logger changed -- batch_bins,
# accumulate_grad_batches and num_device are untouched (see
# tests/test_training_config.py::test_smoke_config_only_differs_from_base_in_five_places),
# because the point is to measure PRODUCTION throughput, not a smaller run.
#
# `#SBATCH --output=logs/smoke_%j.out` requires logs/ to already exist --
# Slurm does not create the parent directory. `mkdir -p logs` before
# submitting.
#
# ---------------------------------------------------------------------------
# Four measurements to extract from the job log after it completes
# (spec section 8, Task 11 step 4):
# ---------------------------------------------------------------------------
#
# 1. UPDATES PER HOUR at production batch_bins.
#    "[train] distributed batches: world_size=..." (dataloader.py) marks
#    when training data is ready; step/time deltas from the MetricsLogger's
#    periodic "Nepoch:train:A-Bbatch: ..." lines (log_every_n_steps: 50)
#    give steps/sec directly. Extrapolate to 1,200,000 updates:
#      wall_clock_hours = 1_200_000 / updates_per_hour
#
# 2. PEAK GPU MEMORY per rank, confirming Base fits on V100-32 in fp32 at
#    this batch_bins. Query after the job completes:
#      sacct -j <jobid> --format=JobID,MaxRSS,MaxVMSize,ReqGRES,Elapsed
#    or (more precise for GPU memory specifically) grep the job's stderr
#    for any CUDA OOM trace (absence is itself the pass condition) and read
#    peak allocated/reserved from PyTorch's own end-of-run summary if
#    trainer logging captures one; otherwise wrap the srun below in
#    `nvidia-smi --query-gpu=memory.used --format=csv -l 30 > logs/gpu_%j.csv &`
#    (backgrounded, killed after srun returns) for a time series.
#
# 3. PEAK HOST RSS and WALL TIME OF SAMPLER CONSTRUCTION, which gates
#    Task 12 (already shipped -- NumElementsArraySampler with max_samples=64
#    is already the config's dataloader.train.iter_factory.batches.type).
#    This still matters for a months-long chained run, which restarts often
#    and pays sampler-construction cost every restart:
#      - Peak RSS per rank: `sacct -j <jobid> --format=JobID,MaxRSS,MaxRSSNode`
#        (per-task RSS is only available per srun step, not per Slurm task,
#        on some PSC accounting configs -- fall back to
#        `/usr/bin/time -v` wrapping the srun invocation below if sacct's
#        MaxRSS reads 0 or blank).
#      - Wall time of sampler construction: the timestamp delta between the
#        LAST "create_shape"/config-resolution log line and the FIRST
#        "[train] distributed batches: world_size=..., rank=..., total=...,
#        per_rank=..." line (espnet3/components/data/dataloader.py). Every
#        log line is timestamped by espnet3/utils/logging_utils.py's
#        LOG_FORMAT, so this is a plain grep + timestamp subtraction, no
#        extra instrumentation needed.
#      - len(batches): read directly off that same "distributed batches"
#        log line (the "total=" field is the full batch count before
#        per-rank slicing).
#
# 4. MP3 DECODE THROUGHPUT (open probe 4): dataloader wait time as a
#    fraction of step time. MetricsLogger already separates iter_time
#    (includes dataloader wait between batches) from forward_time/
#    backward_time/optim_step_time in its periodic summary lines, so
#    iter_time / (iter_time + forward_time + backward_time + optim_step_time)
#    is the dataloader-wait fraction directly from the log, no extra
#    instrumentation needed. If the loader is starving the GPU (this
#    fraction is large), raise dataloader.train.num_workers (currently 4)
#    before considering anything else -- BUT note this node has only 40
#    CPUs total (8 tasks * 5 cpus-per-task already accounts for all of
#    them), so raising num_workers past what's implied by
#    --cpus-per-task requires also raising --cpus-per-task, which lowers
#    how many ranks fit per node.
#
# ---------------------------------------------------------------------------
# Task 12 (numel_array / max_samples=64) success criterion, corrected:
# ---------------------------------------------------------------------------
# The plan's original wording ("confirm peak RSS fell and len(batches) is
# unchanged from the stock run") is UNSATISFIABLE: with max_samples=64
# active, len(batches) MUST increase, because the cap splits the short end
# of the length-sorted order into more, smaller batches. Corrected
# criterion, from the four measurements above:
#   RSS fell; batch count increased only at the short end; no batch
#   exceeds 64.
#
# ---------------------------------------------------------------------------
# Poll on a log marker, not pgrep (an ssh-wrapped `pgrep -f` matches its own
# `bash -c` wrapper and reports alive forever):
#   ssh psc "grep -q 'Training completed' logs/smoke_<jobid>.out && echo DONE || echo RUN"
# ---------------------------------------------------------------------------

set -euo pipefail
cd /ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
source tools/activate_python.sh
cd egs3/emilia/tts

# Offline CSV logger: espnet3 installs LearningRateMonitor unconditionally,
# and Lightning refuses that callback with logger: false.
srun python run.py --stages train \
  --training_config conf/training_f5_tts_base_smoke.yaml
