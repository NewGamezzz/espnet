#!/bin/bash
# Extraction (48 shard tars -> 16 kHz FLAC), phonemization + manifests,
# token list and analytic feats_shape, on one cpu node. Re-runs are free:
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
cd "$(dirname "$0")/.."
ROOT=$(cd ../../.. && pwd)
export PYTHONPATH=$ROOT:$(pwd):${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
PY=/work/nvme/bbjs/ttrachu/pixi_x86/default/bin/python
export PATH=/work/nvme/bbjs/ttrachu/espeak-ng/bin:$PATH
export LD_LIBRARY_PATH=/work/nvme/bbjs/ttrachu/espeak-ng/lib:${LD_LIBRARY_PATH:-}
mkdir -p logs
STAGES=${STAGES:-"create_dataset create_token_list create_shape"}
$PY run.py --stages $STAGES --training_config conf/training_f5_base_dualprompt.yaml
