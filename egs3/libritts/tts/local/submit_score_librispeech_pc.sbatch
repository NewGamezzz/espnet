#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -n 5
#SBATCH -t 8:00:00
#SBATCH -J ls_pc_score
# Validated submission pattern for faster allocation:
#   sbatch -p GPU-shared,GPU-small --gpus=v100-32:1 ...
# The joint GPU-shared,GPU-small QOS caps -t at 8h, so pass -t 8:00:00 with
# the joint override (the in-file -t above already fits, but do not raise it
# past 8h when combining with the joint partition list).
set -euo pipefail
# Usage: sbatch submit_score_librispeech_pc.sbatch <gen_wav_dir>
# <gen_wav_dir> is normally the flat directory
# submit_infer_librispeech_pc.sbatch prints at the end of its run
# (${inference_dir}/wavs).
GEN_WAV_DIR=$(realpath "$1")
DEV=/ocean/projects/cis210027p/ttrachu/development
LS=/ocean/projects/cis210027p/shared/corpora/librispeech/LibriSpeech/test-clean
export HF_HOME=$DEV/f5_eval_assets/hf_cache
source $DEV/f5_eval_assets/venv_scoring/bin/activate
cd "$DEV/F5-TTS"   # scorer resolves ../checkpoints/UniSpeech/... from CWD
python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g "$GEN_WAV_DIR" -p "$LS" -n 1
python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g "$GEN_WAV_DIR" -p "$LS" -n 1
