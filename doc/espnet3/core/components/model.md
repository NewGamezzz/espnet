---
title: ESPnet3 Model Configuration (Training)
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Model Configuration (Training)

This page explains how `model` and `task` in `train.yaml` map to model
construction for the `train` / `collect_stats` stages.

## Two modes: `task` (ESPnet2) vs `model._target_` (custom)

### Use ESPnet2-style models (`task`)

If you want to reuse an ESPnet2-derived model stack, set `task` and use an
ESPnet2-style `model:` block.

```yaml
task: espnet3.systems.asr.task.ASRTask
model:
  encoder: transformer
  decoder: transformer
  # ...ESPnet2-style config...
```

Tip: you can start from existing ESPnet2 configs under `egs2/*/*/conf/*.yaml`.
See the [ESPnet2 task reference](#espnet2-task-reference) for task names and
links to the corresponding recipe docs.

Typical ASR `model:` keys in ESPnet2 configs:

| Key | Purpose |
| --- | --- |
| `encoder` / `encoder_conf` | Encoder type and settings. |
| `decoder` / `decoder_conf` | Decoder type and settings. |
| `model` / `model_conf` | ASR model head and loss settings (CTC/attention, etc.). |
| `frontend` / `frontend_conf` | Feature extraction (e.g., STFT/FBANK). |
| `specaug` / `specaug_conf` | SpecAugment settings. |
| `normalize` / `normalize_conf` | Feature normalization (e.g., global MVN). |

## ESPnet2 task reference

Below is a quick reference to ESPnet2 task names and their recipe docs.

| Task | Description |
| --- | --- |
| [`asr1`](../../../recipe/asr1.md) | Automatic Speech Recognition (Multi-tasking) |
| [`asr2`](../../../recipe/asr2.md) | Automatic Speech Recognition with Discrete Units |
| [`asvspoof1`](../../../recipe/asvspoof1.md) | Speaker Verification Spoofing and Countermeasures |
| [`cls1`](../../../recipe/cls1.md) | Classification |
| [`codec1`](../../../recipe/codec1.md) | Speech Codec |
| [`diar1`](../../../recipe/diar1.md) | Speaker Diarisation |
| [`enh1`](../../../recipe/enh1.md) | Speech Enhancement |
| [`enh_asr1`](../../../recipe/enh_asr1.md) | Speech Recognition with Speech Enhancement |
| [`enh_diar1`](../../../recipe/enh_diar1.md) | Speaker Diarisation with Speech Enhancement |
| [`enh_st1`](../../../recipe/enh_st1.md) | Speech-to-Text Translation with Speech Enhancement |
| [`hubert1`](../../../recipe/hubert1.md) | Self-supervised Learning |
| [`lid1`](../../../recipe/lid1.md) | Language Identification |
| [`lm1`](../../../recipe/lm1.md) | Language Modeling |
| [`mt1`](../../../recipe/mt1.md) | Machine Translation |
| [`s2st1`](../../../recipe/s2st1.md) | Speech-to-Speech Translation |
| [`s2t1`](../../../recipe/s2t1.md) | Weakly-supervised Learning (Speech-to-Text) |
| [`sds1`](../../../recipe/sds1.md) | ESPnet-SDS |
| [`slu1`](../../../recipe/slu1.md) | Spoken Language Understanding |
| [`speechlm1`](../../../recipe/speechlm1.md) | Speech Language Model |
| [`spk1`](../../../recipe/spk1.md) | Speaker Representation |
| [`ssl1`](../../../recipe/ssl1.md) | Self-supervised Learning |
| [`st1`](../../../recipe/st1.md) | Speech-to-Text Translation |
| [`svs1`](../../../recipe/svs1.md) | Singing Voice Synthesis |
| [`svs2`](../../../recipe/svs2.md) | ESPnet2 SVS2 Recipe TEMPLATE |
| [`tts1`](../../../recipe/tts1.md) | Text-to-Speech |
| [`tts2`](../../../recipe/tts2.md) | Text-to-Speech with Discrete Units |
| [`uasr1`](../../../recipe/uasr1.md) | Unsupervised Automatic Speech Recognition |

### Use custom/ESPnet3-only models (`model._target_`)

If you want an ESPnet3-specific or fully custom model, implement it under your
recipe's `src/` directory and point `model._target_` to it:

```yaml
model:
  _target_: src.my_model.MyModel
  # custom args here
```

## Training-time forward contract (common pattern)

For ASR-style training, the training wrapper typically expects your model to
accept batch fields such as:

- `speech`, `speech_lengths`, `text`, `text_lengths`

and return a tuple:

- `loss`: scalar tensor
- `stats`: dict of scalars (logging only)
- `weight`: scalar tensor used as batch size for logging

Example:

```python
class MyCustomModel:
    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        loss = ...
        stats = {"loss": loss.detach()}
        weight = speech.new_tensor(speech.shape[0])
        return loss, stats, weight
```

## Collect-stats support (`collect_feats`)

If you want to use `collect_stats`, your model should implement `collect_feats()`.
See:

- Stage doc: `doc/vuepress/src/espnet3/stages/collect-stats.md`
- Config doc: `doc/vuepress/src/espnet3/config/train_config.md`
