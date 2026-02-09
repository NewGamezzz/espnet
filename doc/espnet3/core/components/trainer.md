---
title: ESPnet3 Trainer Configuration (Training)
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Trainer Configuration (Training)

ESPnet3 uses the PyTorch Lightning Trainer for the `train` stage. Most fields
under `trainer:` in `train.yaml` map directly to Lightning Trainer arguments.

Upstream reference:

- [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

## Common settings

| Key | Description |
| --- | --- |
| `accelerator` | Backend selection (e.g., `auto`, `gpu`, `cpu`). |
| `devices` | Number of devices to use. |
| `num_nodes` | Number of nodes for distributed training. |
| `max_epochs` | Total training epochs. |
| `accumulate_grad_batches` | Gradient accumulation steps. |
| `gradient_clip_val` | Gradient clipping value. |
| `log_every_n_steps` | Logging frequency (steps). |
| `check_val_every_n_epoch` | Validation interval (epochs). |
| `precision` | Mixed precision setting (e.g., `32`, `bf16-mixed`). |
| `strategy` | Distributed strategy (e.g., `ddp`, `auto`). |
| `logger` | Logger configuration (TensorBoard, CSV, etc.). |
| `callbacks` | Training callbacks (checkpointing, averaging, etc.). |

ESPnet3 provides default callbacks. For details and extension points, see:

- `doc/vuepress/src/espnet3/core/components/callbacks.md`

## Example

```yaml
trainer:
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
  max_epochs: 10
  log_every_n_steps: 100
  gradient_clip_val: 1.0
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger
```

## Where this is used in code

The `trainer` config is passed into ESPnet3's training wrapper, which
constructs a `lightning.pytorch.Trainer` internally.

See:

- `espnet3/components/training/trainer.py`
- `doc/vuepress/src/espnet3/config/train_config.md`
