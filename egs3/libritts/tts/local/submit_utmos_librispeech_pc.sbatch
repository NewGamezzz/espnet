#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -n 5
#SBATCH -t 2:00:00
#SBATCH -J ls_pc_utmos
# Validated submission pattern for faster allocation:
#   sbatch -p GPU-shared,GPU-small --gpus=v100-32:1 ...
# The joint GPU-shared,GPU-small QOS caps -t at 8h; the in-file -t above
# already fits.
set -euo pipefail
# Usage: sbatch submit_utmos_librispeech_pc.sbatch <wav_dir> [<wav_dir> ...]
# Each <wav_dir> is a flat directory of wavs, normally the same directory
# passed to submit_score_librispeech_pc.sbatch (${inference_dir}/wavs or
# data/librispeech_pc/gt_wavs). Runs the official F5-TTS eval_utmos.py
# (UTMOS22 strong learner via torch.hub) and writes _utmos_results.jsonl
# into each directory, alongside the _wer/_sim jsonls.
if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 <wav_dir> [<wav_dir> ...]" >&2
  exit 1
fi
DEV=/ocean/projects/cis210027p/ttrachu/development
export HF_HOME=$DEV/f5_eval_assets/hf_cache
# torch.hub cache kept off the tightly quota'd $HOME.
export TORCH_HOME=$DEV/f5_eval_assets/torch_hub
DIRS=()
for d in "$@"; do
  DIRS+=("$(realpath "$d")")
done
source $DEV/f5_eval_assets/venv_scoring/bin/activate
cd "$DEV/F5-TTS"
for d in "${DIRS[@]}"; do
  echo "=== UTMOS: $d ==="
  python src/f5_tts/eval/eval_utmos.py --audio_dir "$d"
done
