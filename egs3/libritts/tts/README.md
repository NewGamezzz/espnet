# ESPnet3 LibriTTS TTS recipe

Two models share this recipe:

- **F5-TTS** (default), a flow-matching non-autoregressive TTS model with
  zero-shot voice cloning from a reference utterance.
- **VITS**, multi-speaker English TTS with x-vector speaker conditioning.

Every stage runs through `run.py`.
There are no cluster submission scripts; adapt the commands below to your own
scheduler.

## F5-TTS

### 1. Prepare data and train

```bash
# Download LibriTTS and build per-split TSV manifests (run once)
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

`compute_xvectors` is not needed for F5-TTS; it belongs to the VITS path only.
For the larger model, substitute `conf/training_f5_tts.yaml` in the training
commands above.
That substitution only changes what gets trained: it does not change which
architecture the inference configs below rebuild, since each of them pins its
own `model.train_config` regardless of what `--training_config` you pass at
inference time.
See the limitation note at the end of Section 3.

### 2. Build the LibriSpeech-PC eval manifest

The default eval set is LibriSpeech-PC test-clean cross-sentence, the protocol
used by arXiv 2410.06885: 1127 same-speaker prompt/target pairs.
It needs two external inputs, the pair list from the F5-TTS repo and a
LibriSpeech `test-clean` tree:

```bash
python local/prepare_librispeech_pc.py \
    --lst <path>/librispeech_pc_test_clean_cross_sentence.lst \
    --test_clean_root <path>/LibriSpeech/test-clean \
    --out_tsv data/librispeech_pc/manifest.tsv
```

### 3. Synthesize

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
Its `model.train_config` is hardcoded to `conf/training_f5_tts.yaml`, the base
architecture, so it expects a checkpoint trained with that config, not the
default small model from Section 1.

To generate the same LibriSpeech-PC set from the official pretrained
`F5TTS_Base` checkpoint as a harness sanity check, run:

```bash
python run.py --stages infer --inference_config conf/inference_pretrained_f5.yaml
```

Unlike the other F5-TTS inference configs, `conf/inference_pretrained_f5.yaml`
sets its own non-empty `exp_tag` (`eval_librispeech_pc_pretrained_base`), so it
does not need `--training_config`.

**Limitation:** `--training_config` only propagates `exp_tag`, `exp_dir`, and
`inference_dir` into the inference config (`espnet3/utils/run_utils.py`'s
`_TRAINING_CONTEXT_KEYS`); it never overrides `model.train_config`.
`conf/inference_f5.yaml` is pinned to `conf/training_f5_tts_small.yaml` (small);
`conf/inference_f5_libritts.yaml` is pinned to `conf/training_f5_tts.yaml`
(base). As shipped, each inference config rebuilds one fixed architecture, so
running a given eval protocol against the other model size means editing that
inference config's `model.train_config` field yourself, not passing a
different `--training_config`. Check this before you burn a GPU hour on a
checkpoint that will fail to load.

### 4. Score

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

## VITS

`conf/metrics.yaml` scores the `librispeech_pc` test set by default.
Before running `measure` below, edit its `dataset.test` list to name the
`valid` and `test` sets instead, or the command will score the wrong split.

```bash
python run.py --stages create_dataset      --training_config conf/training.yaml
python run.py --stages compute_xvectors    --training_config conf/training.yaml
python run.py --stages remove_long_short   --training_config conf/training.yaml
python run.py --stages create_token_list   --training_config conf/training.yaml
python run.py --stages collect_stats       --training_config conf/training.yaml
python run.py --stages train               --training_config conf/training.yaml

python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# Edit conf/metrics.yaml's dataset.test to valid/test before running this.
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
