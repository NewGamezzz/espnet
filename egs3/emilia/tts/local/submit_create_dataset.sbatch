#!/bin/bash
#SBATCH --job-name=emilia_create_dataset
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=logs/create_dataset_%j.out
#
# Builds the full-corpus manifests: ~2,060 shard dirs, 38,105,568 utterances.
#
# WHY A WHOLE RM NODE. The work is 38.1M individual JSON reads off Lustre, so
# it is metadata-LATENCY bound rather than CPU or bandwidth bound, which is
# the regime that parallelises almost linearly. A 16-worker run measured
# 78 utt/s (~4.9 utt/s per worker); 120 workers should land in the 15-20 hour
# range rather than the ~5.5 days 16 workers would need. RM nodes are 128
# cores / 240 GB.
#
# EMILIA_BUILD_WORKERS is 120, not 128: the parent process does the
# accumulate-and-write and should not be fighting its own pool for cores.
#
# MEMORY. dataset/builder.py accumulates EVERY row in the parent before
# writing anything (rows.extend per shard, then one shuffle, then the split).
# At ~450 bytes/row that is roughly 19 GB resident at peak, comfortably inside
# a 240 GB node -- but it also means NOTHING IS WRITTEN UNTIL THE END. A job
# killed on walltime loses all of it. That is why the builder now logs
# "create_dataset: N/M shards | ... | ETA X h" every 1% of shards: check the
# ETA in the first ten minutes, and resubmit larger if it does not fit.
#
# `#SBATCH --output=logs/create_dataset_%j.out` needs logs/ to exist first.
#
# ---------------------------------------------------------------------------
# What this produces, and what must follow it
# ---------------------------------------------------------------------------
#   data/manifest/{train.tsv,valid.tsv}   the manifests
#   data/manifest/shards.txt              shard table (index -> LANG/DIR)
#   data/manifest/report.json             kept/dropped by reason + duration
#                                         histogram
#
# Then create_shape must run before training: it writes exp/stats/*/feats_shape
# analytically from the manifest durations, which the numel sampler needs. It
# is fast (no audio decoding) and is stage 2 of run.py.
#
# ---------------------------------------------------------------------------
# Sanity checks on the output, against upstream's own published numbers
# ---------------------------------------------------------------------------
# src/f5_tts/train/datasets/prepare_emilia.py records upstream's run as:
#     samples count       37837916   (after removal)
#     total duration      95281.87   (hours)
#     bad zh asr cnt        230435   (samples)
#     bad eh asr cnt         37217   (samples)
# and the staged corpus is exactly its 38,105,568 pre-filter total (verified
# by counting every shard, job 43746455). So after this stage:
#   * kept should be near 37.84M, NOT 36.9M -- the collaborator's older
#     manifests hold 36,945,648 because their prep over-filtered by ~4.3x.
#   * report.json's dropped-by-reason should put ZH near 230k and EN near 37k.
#     A ZH figure near zero means the repetition filter regressed to the
#     word-ngram version that was a no-op on space-free Chinese.
#   * total_hours should be near 95,282.

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

# rjieba is absent from the shared pixi env and F5PinyinPreprocessor imports it
# lazily; pylibs/ is an isolated copy so the shared env stays untouched.
export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"

export EMILIA_BUILD_WORKERS=120

echo "python: $(which python)"
echo "workers: $EMILIA_BUILD_WORKERS  cores: $SLURM_CPUS_PER_TASK"
date

srun python -u run.py --stages create_dataset \
  --training_config conf/training_f5_tts_base.yaml

echo "CREATE_DATASET_EXIT=$?"
date
echo "=== report.json ==="
cat data/manifest/report.json || true
echo
wc -l data/manifest/train.tsv data/manifest/valid.tsv || true
echo CREATE_DATASET_DONE
