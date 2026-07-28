#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -n 5
#SBATCH -t 12:00:00
#SBATCH -J ls_pc_infer
set -euo pipefail
DEV=/ocean/projects/cis210027p/ttrachu/development
RECIPE=$DEV/espnet_proj/espnet_f5_eval/egs3/libritts/tts
CONFIG=${CONFIG:-conf/inference_f5_librispeech_pc.yaml}
export HF_HOME=$DEV/f5_eval_assets/hf_cache
export PYTHONPATH=$DEV/espnet_proj/espnet_f5_eval:$RECIPE
source $DEV/f5_eval_assets/venv_infer/bin/activate
cd $RECIPE

# Real entry point (read from run.py in this recipe): stages are selected via
# --stages, and a standalone inference config needs no --training_config as
# long as it sets its own exp_tag (ours does) -- see
# espnet3/utils/run_utils.py's validate_experiment_context/_has_exp_identity.
python run.py --stages infer --inference_config "$CONFIG"

# The infer stage (espnet3/systems/base/inference.py:`infer()`, via
# `_materialize_output_value`) writes wav artifacts to
# ${inference_dir}/<test_set>/<field>/<utt_id>.wav -- for this config that is
# ${inference_dir}/librispeech_pc/wav/<utt_id>.wav, NOT the flat
# <inference_dir>/wavs/<utt_id>.wav the official F5-TTS scorer expects.
# Resolve inference_dir the same way run.py does, then symlink-collect into a
# flat directory for submit_score_librispeech_pc.sbatch.
INFERENCE_DIR=$(python -c "
from pathlib import Path
from espnet3.utils.config_utils import load_and_merge_config
cfg = load_and_merge_config(Path('$CONFIG'), config_name='inference.yaml')
print(cfg.inference_dir)
")
INFERENCE_DIR=$(realpath "$INFERENCE_DIR")
SRC_WAV_DIR="$INFERENCE_DIR/librispeech_pc/wav"
GEN_WAV_DIR="$INFERENCE_DIR/wavs"

shopt -s nullglob
wav_files=("$SRC_WAV_DIR"/*.wav)
if [ ${#wav_files[@]} -eq 0 ]; then
    echo "No generated wavs found under $SRC_WAV_DIR" >&2
    exit 1
fi
mkdir -p "$GEN_WAV_DIR"
for f in "${wav_files[@]}"; do
    ln -sf "$f" "$GEN_WAV_DIR/$(basename "$f")"
done
echo "Flat generated-wav dir for scoring: $GEN_WAV_DIR (${#wav_files[@]} files)"
