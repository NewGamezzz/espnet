#!/bin/bash
# Acceptance checks for the character manifests of the stock F5-TTS baseline
# row S0a (data_chars/) against the phone manifests (data/) built from the
# same packs and rows. Run by local/build_chars_manifest.sbatch after the
# build, or alone from the recipe dir:
#   sbatch --account=bbjs-delta-cpu --partition=cpu --cpus-per-task=8 \
#     --mem=64G --time=01:00:00 --wrap "bash local/verify_chars_manifest.sh"
# Expected: the chars side may hold a few MORE rows (the YODAS rows whose
# transcript eSpeak spelled out into hundreds of phones are short as
# characters: ru 50, es 9, fr 9, id 2, it 1 on 2026-09-29), never zh rows and
# never rows only on the phones side; on the common rows columns 1,2,4-9
# (utt_id, audio, lang, source, group, dur, jsonl_path, byte_offset) are
# identical. Columns 10-12 (speaker mode, word bounds, split targets) may
# differ: a split point is kept only when its tail fits the row, and tails
# count differently in characters than in phones.
# No pipeline here feeds head: under pipefail a SIGPIPE'd cut exits 141 and
# Slurm marks the job failed (job 23191988).
set -euo pipefail
export LC_ALL=C
PH=${1:-data}
CH=${2:-data_chars}
echo "== rows"
for split in train valid; do
  echo "$split: phones $(wc -l < "$PH/manifest/$split.tsv" | tr -d " ")  chars $(wc -l < "$CH/manifest/$split.tsv" | tr -d " ")"
done
echo "== utt_id symmetric difference (train): $(comm -3 <(cut -f1 "$PH/manifest/train.tsv" | sort) <(cut -f1 "$CH/manifest/train.tsv" | sort) | wc -l | tr -d " ")"
echo "== rows only in phones (train): $(comm -23 <(cut -f1 "$PH/manifest/train.tsv" | sort) <(cut -f1 "$CH/manifest/train.tsv" | sort) | wc -l | tr -d " ")"
echo "== rows only in chars (train), per language:"
comm -13 <(cut -f1 "$PH/manifest/train.tsv" | sort) <(cut -f1 "$CH/manifest/train.tsv" | sort) | cut -c1-2 | sort | uniq -c
# "|" is a character for stock F5 (job 23160666 failed on it before the
# tokenizer kept it); report how many rows carry it and where
echo "== rows whose text holds a '|' token (train), per language:"
awk -F'\t' '{n=split($3,a," "); for(i=1;i<=n;i++) if(a[i]=="|"){print substr($1,1,2); break}}' "$CH/manifest/train.tsv" | sort | uniq -c
echo "== columns 1,2,4-9 on common rows (train):"
# restrict the chars side to the phone side's utt_ids, then compare
if cmp <(cut -f1,2,4-9 "$PH/manifest/train.tsv" | sort) \
       <(awk -F'\t' 'NR==FNR{k[$1]; next} ($1 in k)' "$PH/manifest/train.tsv" "$CH/manifest/train.tsv" | cut -f1,2,4-9 | sort); then
  echo "identical"
else
  echo "DIFFER: a common row changed outside the text and split columns; inspect with comm"
  exit 1
fi
echo "== tokens: $(wc -l < "$CH/tokens/tokens.txt" | tr -d " ") lines, specials present: $(grep -cE "^(<space>|<spk>|<lang>|<unk>|<blank>|<sos/eos>)$" "$CH/tokens/tokens.txt") of 6"
echo "== lang_stats"; cat "$CH/lang_stats.json"
echo "== sample lanes (valid): first two, first zh"
head -2 "$CH/manifest/valid.tsv" | cut -f3 | cut -c1-200
awk -F'\t' '$4 == "zh" {print substr($3, 1, 200); exit}' "$CH/manifest/valid.tsv"
awk -F'\t' '{n=split($3,a," "); for(i=1;i<=n;i++) if(a[i]=="|"){print substr($3, 1, 200); exit}}' "$CH/manifest/train.tsv"
echo "== verify done"
