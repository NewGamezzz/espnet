#!/bin/bash
#SBATCH --job-name=emilia_nccl_diag
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=logs/nccl_diag_%j.out
#
# One-shot diagnostic for the hang in job 43826521:
#   Watchdog caught collective operation timeout:
#   WorkNCCL(SeqNum=456, OpType=ALLREDUCE, NumelIn=2201700) ran for 1800027 ms
#   PG status: last enqueued work: 502, last completed work: 455
# on ALL FOUR ranks, after the sanity check passed and training had begun.
#
# DELIBERATELY NOT local/submit_train.sbatch: this must not chain. That script
# queues its successor BEFORE srun so walltime kills continue the run, which is
# right for production and wrong here -- a systematic failure cascades, as it
# did when 43825663 burned as a 24-second successor.
#
# What the extra instrumentation buys:
#   NCCL_DEBUG=INFO            ranks, rings and the transport actually chosen
#   TORCH_NCCL_TRACE_BUFFER_SIZE  enables the flight recorder: a ring buffer of
#                              every collective per rank
#   TORCH_NCCL_DUMP_ON_TIMEOUT dumps that buffer when the watchdog fires, which
#                              is what distinguishes the two live hypotheses:
#     (a) COLLECTIVE ORDER MISMATCH -- ranks disagree about what op #456 is.
#         Expected if _sync2skip's per-step barrier/all_gather
#         (lightning_module.py:207) interleaves with the DDP bucket allreduces
#         that accumulate_grad_batches=8 defers and then flushes together.
#         Signature: different op types or sizes at the same seq across ranks.
#     (b) TRANSPORT STALL -- all ranks agree on op #456 and it simply never
#         completes. Signature: identical entries, all state=scheduled.
#
# NCCL_DEBUG=INFO is extremely verbose at init, so it goes to per-rank files
# rather than drowning the job log.

set -euo pipefail
ROOT=/ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
cd "$ROOT"
set +eu
source tools/activate_python.sh
set -eu
cd "$ROOT/egs3/emilia/tts"

export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

DIAG="$ROOT/diag/$SLURM_JOB_ID"
mkdir -p "$DIAG"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV,GRAPH
export NCCL_DEBUG_FILE="$DIAG/nccl_rank_%h_%p.log"
export TORCH_NCCL_TRACE_BUFFER_SIZE=8192
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DIAG/nccl_trace_rank"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "python : $(which python)"
echo "diag   : $DIAG"
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun python -u run.py --stages train \
  --training_config conf/training_f5_tts_base.yaml || true

echo "DIAG_EXIT=$?"
date
echo "=== flight recorder dumps ==="
ls -la "$DIAG" || true
echo NCCL_DIAG_DONE
