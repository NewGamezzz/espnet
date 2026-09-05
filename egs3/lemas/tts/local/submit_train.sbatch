#!/bin/bash
# Chained training on 4 A100s. The successor is queued BEFORE srun so a
# walltime kill continues the chain; resume picks up exp/<tag>/last.ckpt.
# Usage: sbatch local/submit_train.sbatch [conf/training_....yaml]
#        NO_CHAIN=1 sbatch --time=02:00:00 local/submit_train.sbatch conf/training_smoke.yaml
#SBATCH --job-name=lemas_f5_base
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd ../../.. && pwd)
export PYTHONPATH=$ROOT:$(pwd):${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1
PY=/work/nvme/bbjs/ttrachu/pixi_x86/default/bin/python
CONF=${1:-conf/training_f5_base_dualprompt.yaml}
TAG=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['exp_tag'])" "$CONF")
mkdir -p logs
if [ -z "${NO_CHAIN:-}" ]; then
  sbatch --dependency=afterany:$SLURM_JOB_ID "$0" "$CONF" >> logs/chain.log
fi
CKPT=exp/$TAG/last.ckpt
EXTRA=""
if [ -f "$CKPT" ]; then EXTRA="--ckpt_path $CKPT"; fi
srun $PY run.py --stages train --training_config "$CONF" $EXTRA
