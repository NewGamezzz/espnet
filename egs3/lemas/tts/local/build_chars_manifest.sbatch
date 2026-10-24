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
echo "== rows"
for split in train valid; do
  echo "$split: phones $(wc -l < data/manifest/$split.tsv)  chars $(wc -l < data_chars/manifest/$split.tsv)"
done
echo "== utt_id symmetric difference (train): $(comm -3 <(cut -f1 data/manifest/train.tsv | sort) <(cut -f1 data_chars/manifest/train.tsv | sort) | wc -l)"
echo "== rows only in phones (train): $(comm -23 <(cut -f1 data/manifest/train.tsv | sort) <(cut -f1 data_chars/manifest/train.tsv | sort) | wc -l)"
# the rows only on the chars side must be the text_fits drops of the phone
# build (YODAS rows whose transcript eSpeak spelled out into hundreds of
# phones: ru 50, es 9, fr 9, id 2, it 1 on 2026-09-29), never zh rows: the
# drop_text_regex is applied before tokenization in both builds
echo "== rows only in chars (train), per language:"
comm -13 <(cut -f1 data/manifest/train.tsv | sort) <(cut -f1 data_chars/manifest/train.tsv | sort) | cut -c1-2 | sort | uniq -c
echo "== columns 1,2,4-9 on common rows (train):"
if cmp <(sort -k1,1 data/manifest/train.tsv | cut -f1,2,4-9 | sort) \
       <(sort -k1,1 data_chars/manifest/train.tsv | cut -f1,2,4-9 | sort) ; then
  echo "identical"
else
  echo "differ (expected only on the rows one side lacks; inspect with comm)"
fi
echo "== tokens: $(wc -l < data_chars/tokens/tokens.txt) lines"
grep -c "^<space>$\|^<spk>$\|^<lang>$\|^<unk>$\|^<blank>$\|^<sos/eos>$" data_chars/tokens/tokens.txt
echo "== lang_stats"; cat data_chars/lang_stats.json
echo "== sample lanes"; cut -f3 data_chars/manifest/valid.tsv | head -2 | cut -c1-200
awk -F'\t' '$4 == "zh"' data_chars/manifest/valid.tsv | head -1 | cut -f3 | cut -c1-200
