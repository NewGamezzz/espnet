#! /usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=-1
stop_stage=1
nj=32
val_ratio=0.01
eval_set="eval"

log "$0 $*"
. ../utils/parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ../db.sh || exit 1;

if [ -z "${EMILIA}" ]; then
    log "Fill the value of 'EMILIA' of db.sh"
    exit 1
fi
if [ -z "${SEED_TTS_EVAL}" ]; then
    log "Fill the value of 'SEED_TTS_EVAL' of db.sh"
    exit 1
fi
if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi
if [ -z "${DATA_DIR}" ]; then
    log "Fill the value of 'DATA_DIR' of db.sh"
    exit 1
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    ./data_download.sh "${EMILIA}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    echo "DEBUG data.sh: EMILIA=${EMILIA}"
    echo "DEBUG data.sh: SEED_TTS_EVAL=${SEED_TTS_EVAL}"
    echo "DEBUG data.sh: LIBRISPEECH=${LIBRISPEECH}"
    echo "DEBUG data.sh: DATA_DIR=${DATA_DIR}"
    ./data_prep.sh \
        --nj        "${nj}"        \
        --val_ratio "${val_ratio}" \
        --eval_set  "${eval_set}"  \
        "${EMILIA}/emilia" \
        "${SEED_TTS_EVAL}" \
        "${LIBRISPEECH}"   \
        "${DATA_DIR}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Archive processed data"

    archive_dir="${DATA_DIR}/archives"
    mkdir -p "${archive_dir}"

    # EN training manifests + token list + duration.json
    log "Archiving EN data..."
    tar -czf "${archive_dir}/emilia_en.tar.gz" \
        -C "${DATA_DIR}" \
        EN \
        token_list \
        duration.json
    log "  → ${archive_dir}/emilia_en.tar.gz  ($(du -sh "${archive_dir}/emilia_en.tar.gz" | cut -f1))"

    # ZH training manifests
    log "Archiving ZH data..."
    tar -czf "${archive_dir}/emilia_zh.tar.gz" \
        -C "${DATA_DIR}" \
        ZH
    log "  → ${archive_dir}/emilia_zh.tar.gz  ($(du -sh "${archive_dir}/emilia_zh.tar.gz" | cut -f1))"

    # Eval sets (Seed-TTS EN/ZH + LibriSpeech test-clean)
    log "Archiving eval data..."
    tar -czf "${archive_dir}/eval.tar.gz" \
        -C "${DATA_DIR}" \
        "eval_en" \
        "eval_zh" \
        "librispeech_test_clean_${eval_set}"
    log "  → ${archive_dir}/eval.tar.gz  ($(du -sh "${archive_dir}/eval.tar.gz" | cut -f1))"

    log "All archives written to ${archive_dir}:"
    du -sh "${archive_dir}"/*.tar.gz
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
