#!/bin/bash
# DeltaAI (aarch64, GH200) environment for this recipe. Sourced by
# local/submit_train_dtai.sbatch; usable interactively from the recipe dir.
#   PY      Thanapat's aarch64 pixi env (torch 2.6 cu126, lightning, soundfile,
#           soxr, wandb) on /work/nvme, entered through pixi's shell hook
#   PYLIBS  the Delta uv --target dir with phonemizer (pure Python, shared
#           /work/nvme); no espeak-ng here, since DeltaAI's home is separate,
#           so only training runs on this cluster, not phonemization/inference
# OMP_NUM_THREADS=1: forked DataLoader workers livelock on aarch64 otherwise.
# Do not load the cluster's cudnn module: it would shadow torch's own cudnn.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIXI_MANIFEST=/work/nvme/bbjs/ttrachu/pixi_env/conversational_f5_stage2
eval "$(/u/ttrachu/.pixi/bin/pixi shell-hook --manifest-path $PIXI_MANIFEST)"
PY=$PIXI_MANIFEST/.pixi/envs/default/bin/python
PYLIBS=/work/nvme/bbjs/ttrachu/pylibs/lemas
export PYTHONPATH=$ROOT:$RECIPE:$PYLIBS:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
