#!/bin/bash
# Delta: build the manifests (create_dataset) and the shape files
# (create_shape) from the speaker-tree corpus. CPU only.
#
# create_dataset is resumable: every finished batch is cached under
# data/manifest/.shard_cache, so a job that hits its walltime loses only the
# batches in flight and a resubmission continues from the cache.
#
# Measured for EN+ZH (206 batches, 37.5M rows kept) with 32 workers and an
# empty cache: 7 min 19 s in all, peak memory 67 GB.
#
# Usage, from the recipe dir (the account is given at submit time):
#   sbatch --account=<cpu account> local/delta/prepare_data.sbatch
#   sbatch --account=<cpu account> local/delta/prepare_data.sbatch conf/<other>.yaml
#SBATCH --job-name=emilia_prepare_data
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare_data_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta/env.sh"
CONF=${1:-conf/training_f5_tts_base.yaml}
export EMILIA_BUILD_WORKERS=${EMILIA_BUILD_WORKERS:-$SLURM_CPUS_PER_TASK}
echo "corpus: $EMILIA_CORPUS_ROOT ($EMILIA_CORPUS_LAYOUT)  workers: $EMILIA_BUILD_WORKERS"
git -C "$ROOT" log -1 --format='tree: %h %s' || true
date
$PY local/delta/check_imports.py "$ROOT"
$PY run.py --stages create_dataset --training_config "$CONF"
date
cat data/manifest/report.json
echo
wc -l data/manifest/train.tsv data/manifest/valid.tsv
$PY run.py --stages create_shape --training_config "$CONF"
date
wc -l exp/stats/train/feats_shape exp/stats/valid/feats_shape
echo PREPARE_DATA_DONE
