#!/bin/bash
# DeltaAI: chained training on 2 GH200s in 12 h links (ghx4 allows 2 days and
# was placing jobs within minutes on 2026-10-01). The chain logic lives in
# local/train_link.sh; the aarch64 env in local/delta_ai_env.sh.
# Usage (from delta_ai): sbatch local/submit_train_dtai.sbatch [conf/....yaml]
# Smoke = the first two links of the run, short and unchained:
#   NO_CHAIN=1 sbatch --partition=ghx4-interactive --time=00:20:00 local/submit_train_dtai.sbatch
#   NO_CHAIN=1 sbatch --dependency=afterany:<id> --time=00:20:00 local/submit_train_dtai.sbatch
#SBATCH --job-name=lemas_f5_gh200
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_dtai_%j.out
source "$SLURM_SUBMIT_DIR/local/delta_ai_env.sh"
SELF=submit_train_dtai.sbatch
CHAIN_SUFFIX="_dtai"
CONF_DEFAULT=conf/training_f5_base_dualprompt_gh200.yaml
source "$SLURM_SUBMIT_DIR/local/train_link.sh"
