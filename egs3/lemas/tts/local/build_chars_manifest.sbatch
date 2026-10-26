#!/bin/bash
# Character manifests for the stock F5-TTS baseline row S0a
# (conf/training_f5_base_s0a_stock_chars.yaml): the create_dataset and
# create_token_list stages write data_chars/{manifest,tokens,lang_stats.json}
# from the SAME packs and rows as the phone manifests (the packs are already
# built, so only the manifest build runs). Afterwards the two manifest trees
# are compared: the character manifest may hold a few MORE rows (the YODAS rows
# whose transcript eSpeak spelled out into hundreds of phones are short as
# characters), never different rows for the same utt_id.
# Usage (from the recipe dir on Delta): sbatch local/build_chars_manifest.sbatch
#SBATCH --job-name=lemas_build_chars
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/build_chars_%j.out
set -euo pipefail
# $0 is Slurm's spool copy of this script; the recipe dir is where sbatch ran.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
export PYTHONUNBUFFERED=1
mkdir -p logs
CONF=conf/training_f5_base_s0a_stock_chars.yaml
$PY run.py --stages create_dataset create_token_list --training_config "$CONF"
# acceptance checks (also runnable alone, see the script header)
bash local/verify_chars_manifest.sh data data_chars
