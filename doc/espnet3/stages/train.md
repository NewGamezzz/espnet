---
title: ESPnet3 Train Stage
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Train Stage

The `train` stage runs model training using:

- `espnet3.systems.base.training.train`
- `espnet3.components.modeling.lightning_module.ESPnetLightningModule`
- `espnet3.components.trainers.trainer.ESPnet3LightningTrainer`

## Quick usage

### Run

```bash
python run.py --stages train --training_config conf/training.yaml
```

### Configure (in `training.yaml`)

Keep the core settings in `training.yaml`.

| Section | Description |
| --- | --- |
| `task` | task entrypoint for ESPnet2-style models |
| `model` | model definition and normalization settings |
| `dataset` | train and valid splits |
| `dataloader` | collate and iterator settings |
| `trainer` | Lightning trainer configuration |
| `optimizer`, `scheduler` | single-optimizer training path |
| `optimizers`, `schedulers` | named multi-optimizer path |
| `exp_dir` | training output directory |
| `stats_dir` | stats output directory used by `collect_stats` |

## Main config sections

Training uses `training.yaml`.

Typical sections are:

- `task` or `model`
- `dataset`
- `dataloader`
- `optimizer` / `scheduler` or `optimizers` / `schedulers`
- `trainer`
- `exp_dir`
- `stats_dir`

## Outputs

Training writes under `exp_dir`, including:

- checkpoints
- logs
- TensorBoard output if configured

`collect_stats` writes under `stats_dir`.

Typical outputs are written under:

- `exp_dir`: checkpoints, logs, TensorBoard files, saved configs
- `stats_dir`: feature shapes and normalization stats from `collect_stats`

## Key ideas

### Dataset

Current dataset definitions are based on `DataOrganizer` plus dataset reference
entries using `data_src` and `data_src_args`.

Example:

```yaml
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
```

Details:

- [Training dataset config](./train/dataset.md)
- [Dataset references and builders](../core/components/datasets.md)

### Dataloader

| Topic | Summary | Details |
| --- | --- | --- |
| `dataloader` | supports ESPnet iterator mode and plain PyTorch `DataLoader` mode | [Dataloader + Collate](./train/dataloader.md) |
| `trainer` | uses the ESPnet3 Lightning trainer wrapper, not raw Lightning directly | [Trainer](../core/components/trainer.md) |
| `optimizer`, `scheduler`, `optimizers`, `schedulers` | supports both single-optimizer and named multi-optimizer training | [Optimizer + Scheduler](./train/optim_scheduler.md), [Multiple optimizers and schedulers](../core/components/multiple_optimizers_schedulers.md) |
| `model` | supports both task-backed ESPnet2 model construction and direct Hydra instantiation | [Model](../core/components/model.md) |
| `callbacks` | handles logging, checkpointing, and metric reporting such as `MetricsLogger` | [Callbacks](../core/components/callbacks.md) |

### Details by topic

- [Dataset](./train/dataset.md)
- [Dataloader + Collate](./train/dataloader.md)
- [Trainer](../core/components/trainer.md)
- [Optimizer + Scheduler](./train/optim_scheduler.md)
- [Callbacks](../core/components/callbacks.md)

The train stage is intentionally thin: most customization happens in one of
those component configs rather than in a stage-specific CLI path.

## Related pages

- [Training config](../config/train_config.md)
- [Training dataset config](./train/dataset.md)
- [Dataloader](./train/dataloader.md)
- [Optimizer and scheduler](./train/optim_scheduler.md)
- [Trainer](../core/components/trainer.md)
