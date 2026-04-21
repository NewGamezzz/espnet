---
title: ESPnet3 Optimizer And Scheduler Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Optimizer And Scheduler Configuration

ESPnet3 wraps PyTorch Lightning so that optimizers and schedulers can be
defined from YAML.

Current implementation:

- `espnet3.components.modeling.lightning_module.ESPnetLightningModule.configure_optimizers`
- `espnet3.components.modeling.optimization_spec`

Two modes are supported:

1. single optimizer path: `optimizer` + `scheduler`
2. named multi-optimizer path: `optimizers` + `schedulers`

## What lives in `configure_optimizers` vs YAML

| Layer | You control via YAML | ESPnet3 ensures |
| --- | --- | --- |
| `optimizer` / `optimizers` | optimizer classes and hyperparameters | correct parameter grouping and uniqueness |
| `scheduler` / `schedulers` | scheduler classes and decay settings | matching schedulers to optimizers |
| model parameter names | which parameters each optimizer sees | every trainable parameter is assigned exactly once |

## 1. Single optimizer path

Use `optimizer` and `scheduler` when the whole model shares one optimizer.

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100000

scheduler_interval: step
scheduler_monitor:
```

Important points:

- use `torch.optim.*`, not `torch.optimizer.*`
- `optimizer` is instantiated with all trainable parameters
- `scheduler` is instantiated with that optimizer
- `scheduler_interval` must be `step` or `epoch`
- `scheduler_monitor` is only needed for monitored epoch schedulers such as
  `ReduceLROnPlateau`

Example with a monitored scheduler:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  patience: 2
  factor: 0.5

scheduler_interval: epoch
scheduler_monitor: valid/loss
```

## 2. Named multi-optimizer path

Use `optimizers` and `schedulers` when different parameter groups need
independent updates.

This is the normal path for GAN training.

```yaml
optimizers:
  generator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: generator
    accum_grad_steps: 1
    step_every_n_iters: 1
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm

  discriminator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: discriminator
    accum_grad_steps: 1
    step_every_n_iters: 1

schedulers:
  generator:
    scheduler:
      _target_: torch.optim.lr_scheduler.LinearLR
      start_factor: 1.0
      end_factor: 0.5
      total_iters: 1000
    interval: step

  discriminator:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
      factor: 0.5
    interval: epoch
    monitor: valid/discriminator/loss
```

Important rules:

- names under `optimizers` and `schedulers` must match exactly
- every optimizer entry must include `params` and `optimizer`
- every trainable parameter must match exactly one optimizer
- top-level `scheduler_interval` and `scheduler_monitor` are not used here
- per-optimizer grad settings live under `optimizers.<name>`

## Parameter routing

`params` is a dot-boundary-aware selector over parameter names.

That means the YAML decides which part of the model each optimizer updates.

If parameters are:

- missing from all optimizers
- or matched by more than one optimizer

ESPnet3 raises an error.

## Per-optimizer grad controls

Each named optimizer may define:

- `accum_grad_steps`
- `step_every_n_iters`
- `gradient_clip_val`
- `gradient_clip_algorithm`

These are enforced by ESPnet3's manual optimization path, not by Lightning's
global trainer settings.

## Scheduler stepping rules

Single optimizer path:

- `scheduler_interval: step|epoch`
- optional `scheduler_monitor`

Named multi-optimizer path:

- `schedulers.<name>.interval: step|epoch`
- `schedulers.<name>.monitor`

Step-based schedulers are stepped immediately after that optimizer updates.
Epoch-based schedulers are stepped at epoch end.

## Model-side contract

Single optimizer path expects the model to return a plain tensor loss.

Named multi-optimizer path expects:

- `OptimizationStep`
- or `list[OptimizationStep]`

This is how ESPnet3 knows which optimizer should update.

See [Multiple optimizers and schedulers](./multiple_optimizers_schedulers.md)
for the full model-side contract.

## What not to mix

Do not mix:

- `optimizer` with `optimizers`
- `scheduler` with `schedulers`

ESPnet3 rejects mixed configuration.

## Related pages

- [Multiple optimizers and `OptimizationStep`](./multiple_optimizers_schedulers.md)
- [Training optimizer user guide](../../stages/train/optim_scheduler.md)
- [Training configuration](../../config/train_config.md)
