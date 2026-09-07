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
# Delta: cpu node. Pass 1 packs the 48 tars into one int16 24 kHz .pcm per
# shard (4.8 TB; ~650 members/s per worker, ~1 h). 24 kHz = the model rate:
# 16 kHz sources are upsampled once here (what the loader used to do per
# item), Emilia en/zh keep their bandwidth (24 kHz as is, 32 kHz -> 24 kHz). Pass 2 re-lays each
# language out into ONE chunk-contiguous pack (<lang>/<lang>.pcm + .index.tsv):
# rows of one speaker/recording sit together in chunks of <= 32 (segment
# order), chunks shuffled. Both passes are sequential I/O (hash buckets).
# Why: random access to 30 M small FLAC files on /work/hdd cost ~160 ms per
# open, random 64 KB preads inside a pack 60-75 ms (p90 200-360 ms under
# load), while a 4 MB aligned read costs ~100 ms. The loader therefore reads
# 4 MB blocks and serves a whole batch plus both prompts from one or two of
# them, which needs a speaker's rows and its language-prompt partners to be
# neighbours in the pack. Stripe the root first:
#   lfs setstripe -c 4 -S 4M /work/hdd/bbjs/ttrachu/dataset/LEMAS/poc3k_pcm24k
# phonemizes 30 M rows (zh rows whose text has Latin letters are dropped,
# `drop_text_regex` in dataset/config.yaml; counts land in lang_stats.json),
# writes data/manifest/{train,valid}.tsv,
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
from `--training_config`. Prompts are used as they are (`lowpass_hz: null`):
the training audio is mixed-band (16 kHz sources upsampled, Emilia en/zh at
their native bandwidth), and LEMAS-eval prompts are 16 kHz anyway. The knob
remains for full-band external prompts. Target duration comes from the per-language
tokens-per-second prior in `data/lang_stats.json` times `speed`.

`conf/metrics.yaml` reports, per language: WER (faster-whisper large-v3),
speaker similarity to the speaker prompt, similarity to the language
prompt's voice (the leakage probe, expected low and not rising from arm B
to arm A), and UTMOS. The VERSA dependencies are those of the LibriTTS
recipe (`versa`, `faster-whisper`, `openai-whisper`, `s3prl`).

## Loader throughput

The first smoke (FLAC per file, 4 workers) measured 0.32 s of compute per micro-batch
(batch_bins 1,000,000, 26.6 GB of a 40 GB A100) against 3.1 s of loader wait; shard packs with
12 workers still waited 1.6 s. Decode and resampling cost under 6 ms per item; the rest was
latency-bound random I/O on Lustre, which no worker count fixes. The loader is therefore built
around 4 MB pack blocks:

- `dataset/extract.py` lays each language out chunk-contiguously (speaker chunks of <= 32 rows,
  shuffled), so a row's speaker-prompt partners are its pack neighbours.
- `LEMASDataset` reads whole blocks through a small per-worker cache (`block_samples`,
  `block_cache`) and draws the language prompt from a different speaker in blocks
  b-1..b+1 (`lang_block_span`), falling back to a language-wide draw only inside a giant
  speaker (`n_lang_fallback` counts these).
- `src/sampler.py` (`BlockBatchSampler`, the top-level `batch_sampler`) cuts numel batches
  inside each block, shuffles and shards them per epoch; the plain torch DataLoader path is
  used (`iter_factory: null`, `trainer.use_distributed_sampler: false`,
  `reload_dataloaders_every_n_epochs: 1`). `create_shape` is no longer needed.

Acceptance is `iter_time` at or below `train_time` in the 4-GPU smoke.

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
