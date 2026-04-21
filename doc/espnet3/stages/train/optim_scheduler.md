---
title: ESPnet3 Train Optimizer And Scheduler
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Train Optimizer And Scheduler

This page describes the user-facing config for optimizer and scheduler setup in
`training.yaml`.

The implementation lives in:

- `espnet3.components.modeling.lightning_module.ESPnetLightningModule.configure_optimizers`

## Single optimizer path

Use this for standard training.

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 15000

scheduler_interval: step
scheduler_monitor:
```

Use `scheduler_monitor` only for epoch-based monitored schedulers such as
`ReduceLROnPlateau`.

## Multiple optimizers

Use this for GAN-style or otherwise asymmetric training.

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
      total_iters: 1000
    interval: step

  discriminator:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
    interval: epoch
    monitor: valid/discriminator/loss
```

## What the model must return

Single optimizer path:

```python
return loss_tensor, stats, weight
```

Multiple-optimizer path:

```python
return [
    OptimizationStep(loss=g_loss, name="generator"),
    OptimizationStep(loss=d_loss, name="discriminator"),
], stats, weight
```

The order of the returned list is the update order.

If a named optimizer is not returned for a batch, it is not updated on that
batch.

## Rules

- use `torch.optim.*`
- do not mix `optimizer` with `optimizers`
- do not mix `scheduler` with `schedulers`
- named optimizer and scheduler keys must match exactly
- trainer-level gradient clipping is not supported in multi-optimizer mode

## Related pages

- [Optimizer configuration](../../core/components/optimizer_configuration.md)
- [Multiple optimizers and `OptimizationStep`](../../core/components/multiple_optimizers_schedulers.md)
