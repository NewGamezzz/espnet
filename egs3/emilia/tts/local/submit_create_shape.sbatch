#!/bin/bash
#SBATCH --job-name=emilia_create_shape
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/create_shape_%j.out
#
# Writes exp/stats/{train,valid}/feats_shape, which the numel sampler needs to
# build batches. Must run after create_dataset and before train.
#
# This stage is ANALYTIC: frames = 1 + int(duration * sr) // hop, taken from
# the manifest durations, never by decoding audio. That is the whole point --
# decoding 37.8M utterances to learn frame counts the JSON already records
# would cost roughly 27 days of serial work. Validated against real decoded
# audio on 19,360 utterances: error bounded to at most ONE frame (p99 = 1,
# mean -0.016) with an identical batch count, because
# espnet2/tts/feats_extract/vocoder_mel.py:84 computes the front-end's own
# frame count the same way.
#
# 16 cores for the memory, not the parallelism: EmiliaDataset loads the whole
# manifest columnar (numpy arrays plus one concatenated UTF-8 text buffer),
# which is roughly 7 GB at 37.8M rows, and RM-shared allocates ~1900M/core.
#
# Expect minutes, not hours. If this runs long, something is decoding audio
# that should not be.

set -euo pipefail
ROOT=/ocean/projects/cis210027p/ttrachu/development/espnet_proj/espnet_emilia
cd "$ROOT"
set +eu
source tools/activate_python.sh
set -eu
cd "$ROOT/egs3/emilia/tts"
export PYTHONPATH="$ROOT/pylibs:$ROOT:$ROOT/egs3/emilia/tts:${PYTHONPATH:-}"

echo "python: $(which python)"; date
srun python -u run.py --stages create_shape \
  --training_config conf/training_f5_tts_base.yaml
echo "CREATE_SHAPE_EXIT=$?"; date
wc -l exp/stats/train/feats_shape exp/stats/valid/feats_shape || true
head -2 exp/stats/train/feats_shape || true
echo CREATE_SHAPE_DONE
