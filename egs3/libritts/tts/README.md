# ESPnet3 LibriTTS TTS recipes

TTS on LibriTTS with two model families sharing one data pipeline:

- VITS: multi-speaker English TTS with x-vector speaker conditioning
  (`conf/training.yaml` + `conf/inference.yaml`).
- F5-TTS: flow-matching zero-shot TTS conditioned on a reference prompt
  (`conf/training_f5_tts.yaml` + `conf/inference_f5.yaml`).

Each config is merged over the near-empty shared defaults in
`egs3/TEMPLATE/tts/conf/`, not over another recipe config, so the two
families stay independent.

## VITS quick start

```bash
# 0) Edit configs to set paths.

# 1) Download LibriTTS and build per-split TSV manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract x-vector speaker embeddings (one .pt file per utterance)
python run.py --stages compute_xvectors --training_config conf/training.yaml

# 3) Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training.yaml

# 4) Build the phoneme token list
python run.py --stages create_token_list --training_config conf/training.yaml

# 5) Collect feature statistics (resumable: set collect_stats.num_shards>1)
python run.py --stages collect_stats --training_config conf/training.yaml

# 6) Train VITS
python run.py --stages train --training_config conf/training.yaml

# 7) Synthesize from test text
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 8) Compute the metrics
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## F5-TTS quick start

F5-TTS skips `compute_xvectors` (no speaker embeddings) and tokenizes text
into characters instead of phonemes.
The mel front-end is `feats_extract: vocoder_mel` (Vocos-style log-mel), and
training tracks an EMA copy of the weights via
`espnet3.components.callbacks.ema.EMACallback`.

```bash
# 0) Edit configs to set paths.
#    Set scheduler.total_steps to the planned number of optimizer updates
#    before training; the linear warmup/decay schedule depends on it.

# 1) Download LibriTTS and build per-split TSV manifests (run once, shared with VITS)
python run.py --stages create_dataset --training_config conf/training_f5_tts.yaml

# 2) Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training_f5_tts.yaml

# 3) Build the character token list
python run.py --stages create_token_list --training_config conf/training_f5_tts.yaml

# 4) Collect feature statistics (mel-frame shapes used for numel batching)
python run.py --stages collect_stats --training_config conf/training_f5_tts.yaml

# 5) Train F5-TTS Base
python run.py --stages train --training_config conf/training_f5_tts.yaml

# 6) Zero-shot synthesis with a cross-speaker reference prompt (Vocos vocoder,
#    EMA weights, paper eval protocol: CFG 2.0, sway -1.0, 32 NFE, Euler)
python run.py --stages infer \
    --training_config conf/training_f5_tts.yaml \
    --inference_config conf/inference_f5.yaml

# 7) Compute the metrics
python run.py --stages measure \
    --training_config conf/training_f5_tts.yaml \
    --inference_config conf/inference_f5.yaml \
    --metrics_config conf/metrics.yaml
```

`conf/training_f5_tts_small.yaml` is the F5-TTS Small variant (dim=768,
depth=18) reproducing the paper's LibriTTS setup on 4 GPUs.
See the header comments in that file for the batch-size derivation, an OOM
caveat, and a known limitation when resuming past `scheduler.total_steps`.
