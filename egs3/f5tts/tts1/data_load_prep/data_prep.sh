#!/usr/bin/env bash

# Emilia (EN + ZH) → training data
# Seed-TTS eval + LibriSpeech test-clean → eval data

set -euo pipefail

nj=8
val_ratio=0.01
eval_set="eval"

. ../utils/parse_options.sh || exit 1;

db_emilia=$1
db_seed_tts=$2
db_librispeech=$3
data_dir=$4

if [ $# != 4 ]; then
    echo "Usage: $0 [Options] <emilia-db> <seed-tts-eval-db> <librispeech-db> <data-dir>"
    echo "e.g.: $0 /ocean/.../raw/emilia /ocean/.../seed-tts-eval /ocean/.../LibriSpeech /ocean/.../data"
    echo ""
    echo "Options:"
    echo "    --nj:        number of parallel jobs (default=${nj})."
    echo "    --val_ratio: fraction of Emilia to use as val (default=${val_ratio})."
    echo "    --eval_set:  name suffix for eval dirs (default=${eval_set})."
    exit 1
fi

# ================================================================
# TRAINING DATA: Emilia (EN + ZH)
# ================================================================
echo "Preprocessing Emilia dataset (EN + ZH)..."

if [ ! -e "${data_dir}/EN/train.csv" ] || [ ! -e "${data_dir}/ZH/train.csv" ]; then
    python3 ./prepare_emilia.py \
        --nj        "${nj}"        \
        --val_ratio "${val_ratio}" \
        "${db_emilia}" \
        "${data_dir}"
else
    echo "Emilia already processed. Skipped."
fi

# ================================================================
# EVAL DATA: Seed-TTS + LibriSpeech
# ================================================================
echo "Preprocessing eval sets..."

if [ ! -e "${data_dir}/eval_en/meta.csv" ] || \
   [ ! -e "${data_dir}/eval_zh/meta.csv" ] || \
   [ ! -e "${data_dir}/librispeech_test_clean_${eval_set}/meta.csv" ]; then
    python3 ./prepare_eval.py \
        --eval_set "${eval_set}" \
        "${db_seed_tts}"    \
        "${db_librispeech}" \
        "${data_dir}"
else
    echo "Eval sets already processed. Skipped."
fi

echo "Successfully prepared all eval data."