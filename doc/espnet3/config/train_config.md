---
title: ESPnet3 Training Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Training Configuration

This page describes the current `training.yaml` used by:

```bash
python run.py --stages create_dataset collect_stats train --training_config conf/training.yaml
```

## Minimum required keys

Typical training runs need:

- `task` or `model`
- `dataset`
- `dataloader`
- `trainer`
- `optimizer` and `scheduler`, or `optimizers` and `schedulers`
- `exp_dir`
- `stats_dir`

Common optional sections:

- `create_dataset`
- `tokenizer`
- `parallel`
- `fit`
- `best_model_criterion`

## Config sections overview

| Section | Description |
| --- | --- |
| `num_device`, `num_nodes` | resource counts for training |
| `task` | ESPnet task entrypoint used to build an ESPnet2-style model |
| `recipe_dir`, `data_dir`, `exp_dir`, ... | path scaffold for outputs and cached assets |
| `create_dataset` | dataset builder kwargs used by `create_dataset` |
| `dataset` | train and valid dataset definitions resolved by `DataOrganizer` |
| `tokenizer` | optional tokenizer or text-builder settings |
| `dataloader` | collate, iterator, sampler, and sharding settings |
| `optimizer`, `scheduler`, `optimizers`, `schedulers` | optimization setup |
| `trainer`, `fit` | Lightning trainer arguments and fit-time options |

## Default values

| Key | Default value |
| --- | --- |
| `num_device` | `1` |
| `num_nodes` | `1` |
| `recipe_dir` | `.` |
| `data_dir` | `${recipe_dir}/data` |
| `exp_tag` | `${self_name:}` |
| `exp_dir` | `${recipe_dir}/exp/${exp_tag}` |
| `stats_dir` | `${recipe_dir}/exp/stats` |
| `optimizer._target_` | `torch.optim.Adam` |
| `optimizer.lr` | `0.002` |
| `scheduler._target_` | `espnet2.schedulers.warmup_lr.WarmupLR` |
| `scheduler.warmup_steps` | `15000` |
| `scheduler_interval` | `step` |
| `parallel.env` | `local` |
| `parallel.n_workers` | `1` |


## Typical path scaffold

```yaml
num_device: 1
num_nodes: 1

recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
dataset_dir: /path/to/your/dataset
```

`exp_tag` is important because it participates directly in experiment directory
naming.

By default, TEMPLATE `training.yaml` uses:

```yaml
exp_tag: ${self_name:}
```

That means `exp_tag` defaults to the config filename. For example,
`training_e_branchformer.yaml` resolves to:

```yaml
exp_tag: training_e_branchformer
```

See [Resolvers](./resolvers.md) for `self_name`.

## Core config layout

This section should be read as a user-authored override config, not as the full
TEMPLATE default.

`mini_an4` and `librispeech_100` both keep the default path scaffold from
`egs3/TEMPLATE/asr/conf/training.yaml` and only override the task-specific
parts they need.

```yaml
task: espnet2.tasks.asr.ASRTask

create_dataset:
  recipe_dir: ${recipe_dir}

dataset:
  train:
    - data_src_args:
        split: train
  valid:
    - data_src_args:
        split: valid

tokenizer:
  vocab_size: 5000

dataloader:
  train:
    iter_factory:
      batches:
        type: sorted
        batch_size: 16

optimizer:
  lr: 0.002

scheduler:
  warmup_steps: 15000

trainer:
  log_every_n_steps: 100
  max_epochs: 10
```

This example matches how real recipes usually work:

- default path values usually stay in TEMPLATE:

| Key | Default value |
| --- | --- |
| `recipe_dir` | `.` |
| `data_dir` | `${recipe_dir}/data` |
| `exp_tag` | `${self_name:}` |
| `exp_dir` | `${recipe_dir}/exp/${exp_tag}` |
| `stats_dir` | `${recipe_dir}/exp/stats` |
- local recipes often omit `data_src` and use `${recipe_dir}/dataset/__init__.py`

## `create_dataset`

`create_dataset` is the config block for the `create_dataset` stage.

The values in this block are forwarded to `DatasetBuilder` methods.

```yaml
create_dataset:
  recipe_dir: ${recipe_dir}
  source_dir: ${dataset_dir}
```

See these pages for details:

- [Create dataset stage](../stages/create-dataset.md)
- [Dataset references and builders](../core/components/datasets.md)


## `dataset`

Dataset entries use `DataOrganizer` and dataset references.

```yaml
dataset:
  recipe_dir: ${recipe_dir}
  train:
    - data_src_args:
        split: train
    - data_src: egs3.librispeech_100.asr.dataset
      data_src_args:
        split: train-clean-100
  valid:
    - data_src_args:
        split: valid
```

Each dataset entry may resolve by:

- dataset tag
- explicit module path
- omitted `data_src` -> `${recipe_dir}/dataset/__init__.py`

Only `data_src_args` is passed to `Dataset(...)`.

See [Dataset references and builders](../core/components/datasets.md) for
`data_src` details.

## `dataloader`

Two common modes:

1. ESPnet iterator mode through `iter_factory`
2. plain PyTorch DataLoader mode with `iter_factory: null`

See [Dataloader and Collate](../stages/train/dataloader.md) for `iter_factory`
details, supported iterator factories, and full config examples.

Sequence iterator example:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 4
        batch_bins: 4000000
```

`multiple_iterator` in ESPnet2 is not supported in current ESPnet3.

Standard DataLoader example:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory: null
    batch_size: 4
    num_workers: 2
    shuffle: true
  valid:
    iter_factory: null
    batch_size: 4
    num_workers: 2
    shuffle: false
```

## `optimizer` / `scheduler`

Single optimizer path:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 1000

scheduler_interval: step
scheduler_monitor:
```

`scheduler_interval` and `scheduler_monitor` work as follows:

| Tag | Description |
| --- | --- |
| `scheduler_interval: step` | step the scheduler after optimizer updates |
| `scheduler_interval: epoch` | step the scheduler at epoch boundaries |
| `scheduler_monitor` | metric name used by monitored schedulers such as `ReduceLROnPlateau` |

Notes:

- `step` is the common choice for schedulers such as `WarmupLR`
- `epoch` is used when the scheduler should react once per epoch
- `scheduler_monitor` is only needed for schedulers that require a monitored value
- use the same metric key that appears in logs, for example `valid/loss`

Named multi-optimizer path:

See [Multiple optimizers and schedulers](../core/components/multiple_optimizers_schedulers.md)
for the full behavior.

```yaml
optimizers:
  generator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: generator
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm

  discriminator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: discriminator

schedulers:
  generator:
    scheduler:
      _target_: torch.optim.lr_scheduler.LinearLR
      total_iters: 1000
    interval: step

  discriminator:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
    interval: epoch
    monitor: valid/discriminator/loss
```

## `trainer`

`trainer` maps to Lightning trainer construction through
`ESPnet3LightningTrainer`.

Example:

```yaml
trainer:
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
  max_epochs: 10
  log_every_n_steps: 100
```

In multi-optimizer mode, trainer-level gradient clipping should not be used.
See [Multiple optimizers and schedulers](../core/components/multiple_optimizers_schedulers.md)
for details.

## Parallel execution

Minimal local example:

```yaml
parallel:
  env: local
  n_workers: 1
```

Minimal SLURM example:

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```

Parallel execution details are documented here:

- [Provider / Runner](../core/parallel/provider_runner.md)
- [Multi-GPU / multi-node](../core/parallel/multiple_gpu.md)

## Model definition

If `task` is set, ESPnet3 uses the ESPnet2 task-side model definition. This is
the normal way to reuse ESPnet2-style model config blocks.

If you want a custom model, leave `task` unset and instantiate the model
directly via Hydra in `model`.

Example with `task`:

```yaml
task: espnet2.tasks.asr.ASRTask

model:
  frontend: default
  encoder: e_branchformer
  decoder: transformer
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

In this case, `model` is interpreted as the task-side model config.
This is usually the copy-and-adapt path from an ESPnet2 recipe config.

Example without `task`:

```yaml
task:

model:
  _target_: my_project.models.MyASRModel
  vocab_size: 5000
  hidden_size: 256
```

In this case, `model._target_` is required because ESPnet3 instantiates the
model directly through Hydra.

## `fit`

`training_config.fit` is forwarded to `trainer.fit(...)`.

```yaml
fit: {}
```

This is where runtime fit-time overrides belong.

Resume example:

```yaml
fit:
  ckpt_path: ${exp_dir}/last.ckpt
```

This resumes training from the given checkpoint.

## Related pages

- [Train stage](../stages/train.md)
- [Create dataset stage](../stages/create-dataset.md)
- [Dataset references and builders](../core/components/datasets.md)
- [Optimizer configuration](../core/components/optimizer_configuration.md)
