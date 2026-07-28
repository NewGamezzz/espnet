#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -n 5
#SBATCH -t 8:00:00
#SBATCH -J ls_pc_score
set -euo pipefail
# Usage: sbatch submit_score_librispeech_pc.sbatch <gen_wav_dir>
# <gen_wav_dir> is normally the flat directory
# submit_infer_librispeech_pc.sbatch prints at the end of its run
# (${inference_dir}/wavs).
GEN_WAV_DIR=$1
DEV=/ocean/projects/cis210027p/ttrachu/development
LS=/ocean/projects/cis210027p/shared/corpora/librispeech/LibriSpeech/test-clean
export HF_HOME=$DEV/f5_eval_assets/hf_cache
source $DEV/f5_eval_assets/venv_scoring/bin/activate
cd $DEV/F5-TTS   # scorer resolves ../checkpoints/UniSpeech/... from CWD
python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g "$GEN_WAV_DIR" -p "$LS" -n 1
python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g "$GEN_WAV_DIR" -p "$LS" -n 1
