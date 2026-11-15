#!/bin/bash
# NCSA Delta (x86, A100) environment for this recipe. Sourced by the job
# scripts in this directory and usable interactively from the recipe dir:
#   source local/delta/env.sh
#
# Site settings are NOT kept in version control. Put them in the untracked
# local/delta/site_env.sh (or a file named by EMILIA_SITE_ENV), or export them:
#   EMILIA_PY             python of an env with torch, lightning, soundfile
#                         (mp3-capable libsndfile), vocos, rjieba, pypinyin
#                         and pyworld (imported by espnet2.tasks.tts)
#   PYTHONPATH            optional: a directory of extra packages the env
#                         lacks (pip/uv --target); kept after this tree
#   EMILIA_CORPUS_ROOT    directory holding emilia/<LANG>/...
#   EMILIA_CORPUS_LAYOUT  optional, default speaker_tree (dataset/layout.py)
_DELTA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE=$(cd "$_DELTA_DIR/../.." && pwd)
ROOT=$(cd "$RECIPE/../../.." && pwd)
_SITE_ENV=${EMILIA_SITE_ENV:-$_DELTA_DIR/site_env.sh}
if [ -f "$_SITE_ENV" ]; then
  source "$_SITE_ENV"
fi
: "${EMILIA_PY:?set EMILIA_PY in $_SITE_ENV or the environment}"
: "${EMILIA_CORPUS_ROOT:?set EMILIA_CORPUS_ROOT in $_SITE_ENV or the environment}"
PY=$EMILIA_PY
export EMILIA_CORPUS_ROOT
export EMILIA_CORPUS_LAYOUT=${EMILIA_CORPUS_LAYOUT:-speaker_tree}
# This tree first: the shared env has another espnet checkout installed, and
# espnet3/egs3 must not resolve from there while src/ and dataset/ load here.
export PYTHONPATH=$ROOT:$RECIPE:${PYTHONPATH:-}
# pip --user packages would shadow the env's own and differ per user
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
