#!/bin/bash
# Delta: chained training on 1 A100 in 1 h links, for the layout diagnostics
# (conf/training_f5_small_en_*.yaml). The chain logic lives in
# local/train_link.sh; chains of different configs coexist (keyed by exp_tag).
# Usage: sbatch local/submit_train_1gpu.sbatch conf/training_f5_small_en_....yaml
#        NO_CHAIN=1 sbatch --time=00:30:00 local/submit_train_1gpu.sbatch <config>
#SBATCH --job-name=lemas_f5_diag
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_1gpu_%j.out
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export OMP_NUM_THREADS=4
SELF=submit_train_1gpu.sbatch
CHAIN_SUFFIX="_1gpu"
CONF_DEFAULT=""
source "$SLURM_SUBMIT_DIR/local/train_link.sh"
