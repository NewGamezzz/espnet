#!/bin/bash
# Delta (x86, A100) environment for this recipe. Sourced by the sbatch scripts
# and usable interactively: `source local/delta_env.sh` from the recipe dir.
#   PY      x86 pixi env (torch 2.6 cu124, soundfile, soxr, vocos, lightning)
#   PYLIBS  uv --target dir holding phonemizer (kept out of the shared pixi env)
#   espeak  1.52 built from /work/nvme/bbjs/ttrachu/espeak-ng with CC=gcc and the
#           cudatoolkit module unloaded (the Cray wrappers otherwise link libcupti)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/work/nvme/bbjs/ttrachu/pixi_x86/default/bin/python
PYLIBS=/work/nvme/bbjs/ttrachu/pylibs/lemas
export PYTHONPATH=$ROOT:$RECIPE:$PYLIBS:${PYTHONPATH:-}
export PHONEMIZER_ESPEAK_LIBRARY=/u/ttrachu/.local/lib/libespeak-ng.so
export ESPEAK_DATA_PATH=/u/ttrachu/.local/share/espeak-ng-data
export PATH=/u/ttrachu/.local/bin:$PATH
export PYTHONUNBUFFERED=1
