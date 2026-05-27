---
title: ESPnet3 Callbacks
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Callbacks

ESPnet3's default training callbacks are implemented in:

- `espnet3.components.callbacks.default_callbacks`

The default stack is created by:

- `get_default_callbacks(...)`

## Default callback stack

The current default list is:

- last-checkpoint `ModelCheckpoint`
- best-k `ModelCheckpoint`
- `AverageCheckpointsCallback`
- `LearningRateMonitor`
- `MetricsLogger`
- `TQDMProgressBar`

These are added automatically by `ESPnet3LightningTrainer`.

## MetricsLogger

`MetricsLogger` is the main human-readable training log callback in current
ESPnet3.

Implementation:

- `espnet3.components.callbacks.default_callbacks.MetricsLogger`

<div class='custom-h3'><p>Why it exists</p></div>


Instead of scattering summary logging across multiple callbacks or stage code,
ESPnet3 centralizes compact train/validation summaries in one callback.

<div class='custom-h3'><p>What it logs</p></div>


`MetricsLogger` handles three reporting points:

- interval train-batch summaries
- end-of-epoch train summaries
- end-of-epoch validation summaries

It also tracks timing-style keys such as:

- `iter_time`
- `forward_time`
- `backward_time`
- `optim_step_time`
- `train_time`
- `valid_time`

and includes optimizer learning rates such as `optim0_lr0`.

<div class='custom-h3'><p>Metric key normalization</p></div>


The callback normalizes keys before printing:

- training summaries drop the `train/` prefix
- validation summaries drop the `valid/` prefix
- validation sanity-check runs are ignored

This is why logs stay compact even though the underlying metric names are stored
as `train/...` and `valid/...`.

<div class='custom-h3'><p>Example log lines</p></div>


Typical output looks like:

```text
1epoch:train:1-500batch: loss=12.34 iter_time=0.02 forward_time=0.01 backward_time=0.01 optim_step_time=0.00 train_time=0.03 optim0_lr0=0.0020
epoch_summary:1epoch:train: loss=8.91 iter_time=0.02 forward_time=0.01 backward_time=0.01 optim_step_time=0.00 train_time=0.03 optim0_lr0=0.0020
epoch_summary:1epoch:valid: loss=7.85 cer=18.4 valid_time=12.6
```

## AverageCheckpointsCallback

This callback averages the top-k checkpoint weights chosen by the monitored
`ModelCheckpoint` callbacks and writes:

```text
<monitor>.ave_<K>best.pth
```

Only model weights under `model.*` are exported into the averaged `.pth`.

## Extending callbacks

Custom callbacks can still be appended from config:

```yaml
trainer:
  callbacks:
    - _target_: my_project.callbacks.MyCallback
```

The default stack remains, and custom callbacks are appended after it.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Trainer"
    desc="See how callbacks are attached and used by the trainer wrapper."
    icon="tabler:player-play"
    href="./trainer.md"
  />
  <DocCard
    title="Training configuration"
    desc="See where callback settings live in YAML."
    icon="tabler:settings-2"
    href="../../config/training.md"
  />
  <DocCard
    title="Train stage"
    desc="Return to the stage-level training overview."
    icon="tabler:route"
    href="../../stages/train.md"
  />
</DocCards>
