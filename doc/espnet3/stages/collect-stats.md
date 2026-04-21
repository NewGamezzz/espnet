---
title: ESPnet3 Collect Stats Stage
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Collect Stats Stage

`collect_stats` computes shape files and feature statistics used by later
training steps.

The stage uses `training.yaml`, not a separate config.

## Quick usage

### Run

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
```

This runs `collect_stats` over the `train` and `valid` splits and writes
outputs under `stats_dir/train` and `stats_dir/valid`.

### Configure (in `training.yaml`)

`collect_stats` reads the same `training.yaml` used for training. At minimum:

- `stats_dir` must be set
- `dataset` and `dataloader` define the splits and batching
- `model.normalize_conf.stats_file` often points to the produced stats file

Example:

```yaml
stats_dir: ${exp_dir}/stats

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - name: train
      data_src: mini_an4/asr
      data_src_args:
        split: train
        data_path: ${dataset_dir}
  valid:
    - name: valid
      data_src: mini_an4/asr
      data_src_args:
        split: valid
        data_path: ${dataset_dir}

dataloader:
  train:
    iter_factory:
      batches:
        shape_files:
          - ${stats_dir}/train/feats_shape

model:
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

## What it reads

The stage consumes:

- `dataset`
- `dataloader`
- `model`
- `stats_dir`
- `parallel` when configured

Only `train` and `valid` splits are used.

## Outputs

Typical outputs:

```text
${stats_dir}/
  train/
    feats_shape
    feats_stats.npz
    stats_keys
  valid/
    feats_shape
    feats_stats.npz
    stats_keys
```

Notes:

- `collect_stats` only processes `train` and `valid`; `test` is ignored
- during `collect_stats`, `model.normalize_conf.stats_file` is not read as an
  input source of truth; stats are written under `stats_dir`

## Model requirement

The model must support `collect_feats(...)`.

ESPnet task-backed models already do this. Custom models should provide a
compatible `collect_feats()` implementation returning feature tensors and, when
needed, matching `*_lengths`.

## Developer notes

### What runs under the hood

`collect_stats` builds the model and trainer, then calls the trainer-side
stats-collection path.

The important model contract is `collect_feats(...)`. Task-backed models
already provide this. Custom models should return a dict of tensors keyed by
feature name, plus any `*_lengths` entries needed by the batching logic.

Minimal conceptual example:

```python
class MyCustomModel:
    def collect_feats(self, speech, speech_lengths, **kwargs):
        feats = speech
        feats_lengths = speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
```

This is an ASR-style example, but the same rule applies to any task: return the
features whose statistics should be accumulated, plus lengths when batching
depends on them.

For more background on why these files exist and how they are reused later, see
[Collect stats overview](./collect_stats_description.md).

## Related pages

- [Collect stats overview](./collect_stats_description.md)
- [Training config](../config/train_config.md)
- [Dataloader](./train/dataloader.md)
