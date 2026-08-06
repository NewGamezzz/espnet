# ESPnet3 LibriTTS TTS recipe

This recipe trains and evaluates **F5-TTS**, a flow-matching non-autoregressive
TTS model with zero-shot voice cloning from a reference utterance, on LibriTTS.

Every stage runs through `run.py`.
There are no cluster submission scripts; adapt the commands below to your own
scheduler.

## 1. Prepare data and train

```bash
# Download the corpora and build every manifest (run once)
python run.py --stages create_dataset --training_config conf/training_f5_tts_small.yaml

# Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training_f5_tts_small.yaml

# Build the token list
python run.py --stages create_token_list --training_config conf/training_f5_tts_small.yaml

# Collect feature statistics (resumable: set collect_stats.num_shards>1)
python run.py --stages collect_stats --training_config conf/training_f5_tts_small.yaml

# Train
python run.py --stages train --training_config conf/training_f5_tts_small.yaml
```

`conf/training_f5_tts_small.yaml` is the recipe's only training config and its
default: the F5TTS_Small architecture (dim 768, depth 18, heads 12), targeting
the LibriTTS rows of arXiv 2410.06885 Table 9.

`create_dataset` prepares both the training data and the eval data, so no
manual preparation step is needed anywhere in this recipe. It downloads:

- **LibriTTS** (OpenSLR 60), the five subsets listed in
  `dataset/config.yaml`, and writes `data/manifest/{train,valid,test}.tsv`.
- **LibriSpeech `test-clean`** (OpenSLR **12**, a different corpus) plus the
  cross-sentence pair list from the F5-TTS repo, and writes
  `data/librispeech_pc/manifest.tsv` - the eval manifest that
  `conf/inference_f5.yaml` reads.

Budget for the LibriSpeech side on top of LibriTTS: about 350 MB downloaded
and about 350 MB extracted, so roughly 700 MB, since the tarball is kept in
`downloads/` after extraction (flac barely compresses, so the extracted tree
is about the size of the archive). The pair list is a 220 KB text file pinned
to a specific F5-TTS commit, so the eval set cannot shift under you.

Everything is idempotent: extracted subsets carry a `.complete` marker and the
pair list is skipped when present, so re-running `create_dataset` transfers
nothing.

If the corpora already exist on your cluster, `local/prepare_librispeech_pc.py`
remains available as a standalone CLI for building the eval manifest against a
read-only tree:

```bash
python local/prepare_librispeech_pc.py \
    --lst <path>/librispeech_pc_test_clean_cross_sentence.lst \
    --test_clean_root <path>/LibriSpeech/test-clean \
    --out_tsv data/librispeech_pc/manifest.tsv
```

## 2. Synthesize

```bash
python run.py --stages infer \
    --training_config conf/training_f5_tts_small.yaml \
    --inference_config conf/inference_f5.yaml
```

`--training_config` is required here.
`conf/inference_f5.yaml` leaves `exp_tag` empty so it inherits experiment
identity from the training config, and `run.py` rejects an inference config
with no experiment identity of its own.

To evaluate in-domain on the LibriTTS `valid`/`test` splits with cross-speaker
prompts instead, swap in `conf/inference_f5_libritts.yaml`.
Both inference configs pin the same `conf/training_f5_tts_small.yaml`, so
either one loads a checkpoint from Section 1 without further edits.

If you add a training config for a different architecture, note that
`--training_config` only propagates `exp_tag` and `exp_dir` into the inference
config (`espnet3/utils/run_utils.py`'s `_TRAINING_CONTEXT_KEYS`); it never
overrides `model.train_config`. Point the inference config's own
`model.train_config` at the matching training config, or the checkpoint will
fail to load with a shape mismatch.

## 3. Score

```bash
python run.py --stages measure \
    --training_config conf/training_f5_tts_small.yaml \
    --inference_config conf/inference_f5.yaml \
    --metrics_config conf/metrics.yaml
```

Scoring runs through VERSA and reports WER, speaker similarity, and UTMOS.

The WER metric wraps faster-whisper, which espnet's
`tools/installers/install_versa.sh` does not install: it takes VERSA's
`[audio]` extra only.
Run VERSA's own `tools/install_fwhisper.sh` inside `tools/versa` first.

`conf/metrics.yaml` documents how each metric maps onto the official F5-TTS
scorer, including the one metric that cannot be matched exactly.
In short: WER and UTMOS are equivalent to the official implementations, but
speaker similarity uses an ESPnet-SPK model rather than the official UniSpeech
checkpoint, so SIM values are comparable across your own checkpoints but not
against the numbers published in the paper.
