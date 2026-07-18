#!/bin/bash
# Task 6: start the BagPiper vLLM server, wait for it to become healthy,
# run eval/generate_vllm.py against it, then tear the server down.
#
# Adapted verbatim from the proven /work/nvme/bbjs/ttrachu/ab_test/job.sh
# pattern (A/B test that validated generation on Delta) - server flags,
# health-poll budget, and cleanup trap are the proven values; do not
# "clean up" them away.
#
# Usage:
#   sbatch --export=MANIFEST=/work/nvme/bbjs/ttrachu/bagpiper-eval/manifest_sssd.json,OUT_DIR=/work/nvme/bbjs/ttrachu/bagpiper-eval/gen_sssd \
#     eval/delta/serve_and_generate.sbatch
#
#SBATCH --job-name=bagpiper-serve-generate
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

# --- Delta assets (proven paths - see delta-server-usage-rules) --------
GATE_CLONE=/work/nvme/bbjs/ttrachu/bagpiper-gate
RECIPE_DIR="$GATE_CLONE/egs3/conversational/lm_tts"
CONDA_ENV=/work/nvme/bbjs/ttrachu/miniconda3/envs/bagpiper
SIF=/work/nvme/bbjs/hwang41/qwen3/apptainer/vllm_bagpiper.sif
CKPT=/work/nvme/bbjs/hwang41/bagpiper/model_vllm/bagpiper
export HF_HOME=/work/nvme/bbjs/ttrachu/.cache/huggingface
# NOTE: do NOT set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE here. Empirically
# (Delta, 2026-07-17) exporting them - together with binding only the
# checkpoint dir into the container - suppressed the model's audio-output
# path: the server returned finish_reason=stop with no audio for every
# request (0/20). The proven-good launch from the A/B harness
# (/work/nvme/bbjs/ttrachu/ab_test/job.sh) uses `--bind /work --pwd /work`
# and no offline vars, and produces audio reliably (20/20). The vLLM
# server loads the checkpoint from a local path, so no HF egress is needed
# regardless.

PORT=8000
MODEL_NAME=speechlm-qwen3-8b

# --- Required job params (sbatch --export=MANIFEST=...,OUT_DIR=...) ----
: "${MANIFEST:?MANIFEST must be set via sbatch --export=MANIFEST=...,OUT_DIR=...}"
: "${OUT_DIR:?OUT_DIR must be set via sbatch --export=MANIFEST=...,OUT_DIR=...}"

mkdir -p "$OUT_DIR"
SERVER_LOG="$OUT_DIR/vllm_server.log"

echo "job $SLURM_JOB_ID on $(hostname), manifest=$MANIFEST out_dir=$OUT_DIR"

# --- Cleanup: always kill the server process tree on exit --------------
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "cleanup: stopping vLLM server tree (pid $SERVER_PID)"
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  # Name-based safety net: apptainer can orphan worker processes outside
  # the recorded PID tree (e.g. detached CUDA/multiprocessing workers), so
  # the PID-based kill above is not sufficient on its own.
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
}
trap cleanup EXIT

# --- Start the vLLM server (apptainer, proven flags) --------------------
echo "starting vLLM server on port $PORT (log: $SERVER_LOG)"
apptainer exec --nv --bind /work --pwd /work \
  "$SIF" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 1024 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --enable-prefix-caching \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vLLM server pid $SERVER_PID"

# --- Health poll: liveness-checked, 12 minute budget ---------------------
HEALTH_BUDGET_S=$((12 * 60))
POLL_INTERVAL_S=10
elapsed=0
server_ready=0
while (( elapsed < HEALTH_BUDGET_S )); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server process died during startup - see $SERVER_LOG"
    exit 1
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    server_ready=1
    break
  fi
  sleep "$POLL_INTERVAL_S"
  elapsed=$((elapsed + POLL_INTERVAL_S))
done

if [[ "$server_ready" -ne 1 ]]; then
  echo "vLLM server did not become healthy within ${HEALTH_BUDGET_S}s - see $SERVER_LOG"
  exit 1
fi
echo "vLLM server healthy after ${elapsed}s"

# --- Run the client (stdlib-only network path; any python on the env) --
set +e
cd "$RECIPE_DIR"
PYTHONPATH="$GATE_CLONE:$RECIPE_DIR:${PYTHONPATH:-}" \
  "$CONDA_ENV/bin/python" -m eval.generate_vllm \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR" \
    --port "$PORT" \
    --concurrency "${CONCURRENCY:-4}" \
    --timeout 1200 \
    --max-tokens 12000
CLIENT_EXIT=$?
set -e

echo "eval.generate_vllm exited with code $CLIENT_EXIT"
exit "$CLIENT_EXIT"
