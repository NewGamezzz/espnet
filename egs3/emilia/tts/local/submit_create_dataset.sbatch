#!/bin/bash
#SBATCH --job-name=emilia_create_dataset
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
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
# RESUMABLE. Each worker checkpoints its shard to
# data/manifest/.shard_cache/{idx}.tsv plus a {idx}.json written last, both
# via temp+rename. A run killed on walltime therefore loses only the shards
# in flight: RESUBMITTING THIS EXACT SCRIPT PICKS UP WHERE IT STOPPED and
# logs "resuming, N/M shards already cached". Nothing else to do.
#
# The cache is fingerprinted on corpus_root, langs, duration bounds,
# strict_text_filters, audio_suffix and _FILTER_VERSION. Change any of them
# and the build REFUSES to start rather than silently mixing rows produced
# under different filter semantics -- delete .shard_cache to rebuild.
#
# 72h is the RM QOS ceiling (rmpartition MaxWall=3-00:00:00). With resume,
# overrunning it is an inconvenience rather than a lost run.
#
# MEMORY is no longer a factor: the parent holds only counts during the scan,
# and the final split streams the cached shards straight to disk instead of
# materialising ~38.1M rows (~19 GB).
#
# Watch "create_dataset: N/M shards ... ETA X h" (every 1% of shards) to see
# the real rate early.
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
