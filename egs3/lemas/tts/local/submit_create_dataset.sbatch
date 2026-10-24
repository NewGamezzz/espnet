#!/bin/bash
# Extraction (shard tars -> 24 kHz int16 packs, then the chunk-contiguous
# regroup, each language verified against its shard packs), phonemization +
# manifests and the token list, on one cpu node. Re-runs are free:
# extracted shards carry .complete markers.
#SBATCH --job-name=lemas_create_dataset
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/create_dataset_%j.out
set -euo pipefail
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export PYTHONUNBUFFERED=1
mkdir -p logs
STAGES=${STAGES:-"create_dataset create_token_list"}
$PY run.py --stages $STAGES --training_config conf/training_f5_base_dualprompt.yaml
