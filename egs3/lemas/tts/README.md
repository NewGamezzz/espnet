# ESPnet3 LEMAS TTS recipe: dual-prompt F5-TTS

F5-TTS (F5-Base geometry, from scratch) trained on the LEMAS poc3k subset
(10 languages x 3,000 h) with two transcript-free audio prompts:

- a **speaker prompt**: another utterance of the same speaker (or recording),
  whose voice is cloned;
- a **language prompt**: an utterance of the target language by a different
  speaker, which carries how the language is realized.

The text front-end is phonemes (eSpeak-NG IPA via espnet2's `Phonemizer`,
pinyin initial-final with tones for zh via espnet2's `pypinyin_g2p_phone`).
Design: vault note "Design - LEMAS Dual-Prompt F5 Recipe" (2026-09-05).

## Layout of one training sample

```
speech : [ speaker prompt | language prompt | target ]        24 kHz, one waveform
text   : [ <spk> x frames | <lang> x frames | <de> phones ]   one id per prompt frame
cond_frames = speaker frames + language frames                loss only on [cond_frames, len)
```

Prompt partners are drawn **online** by `dataset/dataset.py` for every row on
every epoch (seeded by `(seed, epoch, row)`); each prompt gets a random
length and a random window. Rows whose source has no speaker id split
themselves at a word boundary (`spk_mode: split`) or train prompt-free
(`spk_mode: none`). Dropout is omission of a region: with both prompts
dropped `cond_frames` is 0 and the row is text-only generation.

The model change is one subclass (`src/model.py`): `DualPromptCFM` masks
`[cond_frames, len)` instead of F5's random span. The DiT and the
`espnet3/systems/tts/f5_tts` package are used unchanged.

## 1. Data, token list, shapes

```bash
# Delta: cpu node, 24 h. Extracts 48 tars to 16 kHz FLAC (2.23 TB, 30 M files;
# the Emilia members of en/zh ship at 24/32 kHz and are resampled with soxr,
# counted as `resampled` in each shard's .coverage.json),
# phonemizes 30 M rows, writes data/manifest/{train,valid}.tsv,
# data/lang_stats.json, data/tokens/tokens.txt and exp/stats/*/feats_shape.
sbatch local/submit_create_dataset.sbatch
# equivalent stages:
python run.py --stages create_dataset create_token_list create_shape \
    --training_config conf/training_f5_base_dualprompt.yaml
```

`create_shape` writes `feats_shape` analytically from manifest durations at
the longest prompt layout (an upper bound), so `collect_stats` is not used.
`remove_long_short` is not used either: the 1 to 20 s target filter is
applied in build.

Paths (mirror, FLAC root, languages, filters) live in `dataset/config.yaml`.

## 2. Train

```bash
NO_CHAIN=1 sbatch --time=02:00:00 local/submit_train.sbatch conf/training_smoke.yaml   # measure memory/throughput first
sbatch local/submit_train.sbatch                                                       # chained 48 h jobs
```

Prompt knobs (`prompt_config` in `conf/training_f5_base_dualprompt.yaml`):

| key | default | meaning |
|---|---|---|
| `spk_prompt_sec` | `[1.0, 6.0]` | speaker prompt length range (s), uniform |
| `lang_prompt_sec` | `[1.0, 6.0]` | language prompt length range (s), uniform |
| `split_frac` | `[0.2, 0.4]` | prompt share of a self-split row |
| `split_min_prompt_sec` | `1.0` | floor for a self-split prompt |
| `spk_neighbor_k` | `8` | recording groups: draw among the k nearest segments |
| `p_drop_spk` | `0.3` | drop the speaker prompt (heavier: it also reveals the language) |
| `p_drop_lang` | `0.1` | drop the language prompt |

## 3. Synthesize and score on LEMAS-eval

```bash
# once: split each eval row into prompt / target clips
python local/prepare_lemas_eval.py \
    --metadata /work/hdd/bbjs/ttrachu/dataset/LEMAS/LEMAS-eval/eval/metadata.jsonl \
    --audio_root /work/hdd/bbjs/ttrachu/dataset/LEMAS/LEMAS-eval/eval \
    --out_dir data/lemas_eval

sbatch local/run_arm_1gpu.sbatch conf/inference_lemas_eval.yaml           # arm A: both prompts
sbatch local/run_arm_1gpu.sbatch conf/inference_lemas_eval_spk_only.yaml  # arm B: speaker prompt only
```

Both configs use the training config for the model block; `exp_tag` comes
from `--training_config`. Prompts are low-passed at 8 kHz before use
(`lowpass_hz`): the training audio is 16 kHz-sourced, so a full-band prompt
is out of distribution. Target duration comes from the per-language
tokens-per-second prior in `data/lang_stats.json` times `speed`.

`conf/metrics.yaml` reports, per language: WER (faster-whisper large-v3),
speaker similarity to the speaker prompt, similarity to the language
prompt's voice (the leakage probe, expected low and not rising from arm B
to arm A), and UTMOS. The VERSA dependencies are those of the LibriTTS
recipe (`versa`, `faster-whisper`, `openai-whisper`, `s3prl`).

## Delta environment

`local/delta_env.sh` is sourced by every sbatch script and works interactively
(`source local/delta_env.sh` from the recipe dir). It sets:

- `PY`: the x86 pixi env `/work/nvme/bbjs/ttrachu/pixi_x86/default` (torch 2.6, soundfile, soxr, vocos, lightning, pytest).
- `PYLIBS`: `/work/nvme/bbjs/ttrachu/pylibs/lemas`, a `uv pip install --target` dir holding `phonemizer`
  so the shared pixi env is not modified. Recreate with
  `uv pip install --python $PY --target $PYLIBS "phonemizer>=3.2"`.
- `PHONEMIZER_ESPEAK_LIBRARY` / `ESPEAK_DATA_PATH`: espeak-ng 1.52 installed under `/u/ttrachu/.local`.
  It was rebuilt from `/work/nvme/bbjs/ttrachu/espeak-ng` with `CC=gcc CXX=g++` and the `cudatoolkit`
  module unloaded: the Cray compiler wrappers otherwise inject `-lcupti -lcudart -lcuda` into the link,
  and the library then fails to load once the CUDA module version changes.

Under sbatch, `$0` is Slurm's spool copy of the script, so every script does `cd "$SLURM_SUBMIT_DIR"`;
submit from the recipe dir.

Repo trap: the top-level `.gitignore` rule `egs*/*/*/data*` also matches this recipe's `dataset/`
directory. Stage new files there with `git add -f egs3/lemas/tts/dataset/<file>.py` (never the bare
directory, which would pick up `__pycache__`), and run `black`/`isort` on explicit paths because they
honour the same ignore.

## Tests

```bash
cd egs3/lemas/tts
PYTHONPATH=../../..:$(pwd) python -m pytest tests -q
```
