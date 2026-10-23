#!/bin/bash
# Data for the layout diagnostics (conf/training_f5_small_en_*.yaml): the
# English lines of the train and valid manifests, then each config's
# feats_shape in its own stats dir.
# Usage (from the recipe dir on Delta): sbatch local/prepare_diag.sbatch
#SBATCH --job-name=lemas_prep_diag
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prepare_diag_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
mkdir -p data/manifest_en
for split in train valid; do
  awk -F'\t' '$4 == "en"' data/manifest/$split.tsv > data/manifest_en/$split.tsv.tmp
  mv data/manifest_en/$split.tsv.tmp data/manifest_en/$split.tsv
  echo "$split: $(wc -l < data/manifest_en/$split.tsv) en rows of $(wc -l < data/manifest/$split.tsv)"
done
for conf in conf/training_f5_small_en_dualprompt.yaml conf/training_f5_small_en_stock.yaml; do
  $PY run.py --stages create_shape --training_config "$conf"
done
