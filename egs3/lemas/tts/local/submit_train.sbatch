#!/bin/bash
# Delta: chained training on 4 A100s in 1 h links. On gpuA100x4 every 4-GPU
# job that started in a 24 h window had a limit of 1 h or less (132 of them,
# median wait 0 h; none between 4 h and 24 h started; measured 2026-09-08).
# The chain logic lives in local/train_link.sh.
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
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1
SELF=submit_train.sbatch
CHAIN_SUFFIX=""
CONF_DEFAULT=conf/training_f5_base_dualprompt.yaml
source "$SLURM_SUBMIT_DIR/local/train_link.sh"
