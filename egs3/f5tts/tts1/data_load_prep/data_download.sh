#!/usr/bin/env bash
set -euo pipefail

db_root=$1

if [ $# != 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi

if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it to your Hugging Face read token, e.g., export HF_TOKEN=hf_..."
    exit 1
fi

START_INDEX=0
END_INDEX=1139
cwd=$(pwd)

# ── Emilia (EN + ZH only) ────────────────────────────────────────────────────
echo "Downloading Emilia corpus (EN and ZH)..."
for lang in EN ZH; do
    if [ ! -e "${db_root}/emilia/${lang}/.complete" ]; then
        mkdir -p "${db_root}/emilia/${lang}"
        cd "${db_root}/emilia/${lang}" || exit 1

        for file_index in $(seq $START_INDEX $END_INDEX); do
            idx=$(printf "%06d" "$file_index")
            filename="${lang}-B${idx}"

            if [ -e "${filename}/.complete" ]; then
                echo "${filename} already exists. Skipped."
                continue
            fi

            data_url="https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/Emilia/${lang}/${filename}.tar"
            echo "Downloading ${data_url} ..."
            if ! wget -c --header="Authorization: Bearer ${HF_TOKEN}" "${data_url}"; then
                echo "Warning: Failed to download ${data_url}. Stopping." >&2
                rm -f "${filename}.tar"
                break
            fi

            mkdir -p "${filename}"
            tar -xvf "${filename}.tar" -C "${filename}"
            touch "${filename}/.complete"
            rm -f "${filename}.tar"
        done

        touch "${db_root}/emilia/${lang}/.complete"
        cd "${cwd}" || exit 1
    else
        echo "Emilia ${lang} already exists. Skipped."
    fi
done

# ── Seed-TTS Eval ────────────────────────────────────────────────────────────
echo "Downloading Seed-TTS eval set..."
if [ ! -e "${db_root}/seed-tts-eval/.complete" ]; then
    mkdir -p "${db_root}/seed-tts-eval"
    cd "${db_root}/seed-tts-eval" || exit 1

    pip install -q gdown

    gdown "https://drive.google.com/uc?id=1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"

    tar -xf seedtts_testset.tar
    rm seedtts_testset.tar

    touch .complete
    cd "${cwd}" || exit 1
    echo "Successfully downloaded and extracted Seed-TTS eval data."
else
    echo "Seed-TTS eval already exists. Skipped."
fi

# ── LibriSpeech test-clean ───────────────────────────────────────────────────
echo "Downloading LibriSpeech test-clean..."
if [ ! -e "${db_root}/LibriSpeech/test-clean" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1
    wget -c https://www.openslr.org/resources/12/test-clean.tar.gz
    tar -xzf test-clean.tar.gz
    rm test-clean.tar.gz
    cd "${cwd}" || exit 1
    echo "Successfully downloaded LibriSpeech test-clean."
else
    echo "LibriSpeech test-clean already exists. Skipped."
fi