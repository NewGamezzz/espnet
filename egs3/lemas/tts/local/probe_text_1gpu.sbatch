#!/bin/bash
# Text-use probe of one backup checkpoint on one A100 (local/probe_text_use.py),
# submitted by local/eval_watch.py for every backup of the runs it watches.
# Usage: sbatch local/probe_text_1gpu.sbatch conf/training_....yaml backup_step5000
#SBATCH --job-name=lemas_probe
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/probe_%j.out
set -euo pipefail
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
CONF=$1
CKPT=$2
TAG=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['exp_tag'])" "$CONF")
$PY local/probe_text_use.py --train_config "$CONF" --ckpt "exp/$TAG/$CKPT.ckpt" \
  --n_batches 100 --out exp/$TAG/probe_text_$CKPT.json
