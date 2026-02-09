---
title: ESPnet3 - Optimiser and Scheduler Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

## ⚙️ ESPnet3: Optimiser and Scheduler Configuration

ESPnet3 wraps PyTorch Lightning so that optimisers and schedulers can be defined
purely from the Hydra configuration.  Two modes are supported by
`espnet3.components.modeling.lightning_module.ESPnetLightningModule.configure_optimizers`:

1. a single optimiser/scheduler pair (`optimizer` + `scheduler`)
2. multiple optimiser/scheduler pairs (`optimizers` + `schedulers`)

The sections below describe both.

### ✅ What lives in `configure_optimizers` vs YAML

| Layer           | You control via YAML                               | ESPnet3 (`ESPnetLightningModule`) ensures            |
| --------------- | -------------------------------------------------- | --------------------------------------------- |
| `optimizer` / `optimizers` blocks   | Which optimiser classes and their hyperparameters | Correct parameter grouping and uniqueness     |
| `scheduler` / `schedulers`  | Scheduler types and decay schedules    | Matching schedulers 1:1 with optimisers       |
| Model parameter names       | Which parts of the model each entry sees | That every trainable param is assigned once |

### 1. Single optimiser

Use `optimizer` and `scheduler` when the entire model shares one optimiser.  The
entries are passed directly to `hydra.utils.instantiate`, so any optimiser or
scheduler from PyTorch (or a custom class) is supported.

```yaml
optimizer:
  _target_: torch.optimizer.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optimizer.lr_scheduler.CosineAnnealingLR
  T_max: 100000
```

No additional wiring is necessary because ESPnet3 instantiates both objects, attaches
the scheduler to the optimiser, and returns them to Lightning.

### 2. Multiple optimisers

When different parts of the model need their own optimiser, switch to `optimizers`
and `schedulers`.  Each entry contains a nested `optimizer` block and a `params`
string that selects parameters whose names contain the substring.

```yaml
optimizers:
  - optimizer:
      _target_: torch.optimizer.Adam
      lr: 0.0005
    params: encoder
  - optimizer:
      _target_: torch.optimizer.Adam
      lr: 0.001
    params: decoder

schedulers:
  - scheduler:
      _target_: torch.optimizer.lr_scheduler.StepLR
      step_size: 5
      gamma: 0.5
  - scheduler:
      _target_: torch.optimizer.lr_scheduler.StepLR
      step_size: 10
      gamma: 0.1
```

Important rules enforced by `configure_optimizers`:

- Do not mix the single- and multi-optimiser modes.  Either use `optimizer` +
  `scheduler` or `optimizers` + `schedulers`.
- Every optimiser entry must include `params` and `optimizer`.
- Each trainable parameter must match exactly one optimiser.  ESPnet3 raises an
  error if parameters are missing or assigned twice.
- The number of scheduler entries must equal the number of optimisers.  They are
  matched by position, so the first scheduler controls the first optimiser, etc.

Under the hood ESPnet3 wraps the instantiated optimisers with
`MultipleOptim`/`MultipleScheduler` so that Lightning sees them as a single
optimiser while still stepping all underlying objects together.
