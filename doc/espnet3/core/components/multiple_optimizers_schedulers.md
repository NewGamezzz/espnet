---
title: Multiple Optimizers and Schedulers in ESPnet3
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🧮 Multiple optimizers and schedulers (developer guide)

This page is for developers who want to:

- implement a custom optimizer / scheduler for a recipe (under `egs3/*/*/src/`)
- understand how `conf/train.yaml` is consumed at runtime
- correctly group parameters for multiple optimizers
- debug common configuration errors quickly


Related: [Optimizer configuration](./optimizer_configuration.md) (config overview).


## 1) Where to write custom optimizers / schedulers

### Recipe-local (recommended)

Put custom code in your recipe:

- `egs3/<recipe>/<task>/src/optimizers.py`
- `egs3/<recipe>/<task>/src/schedulers.py`

and reference it from config with Hydra `_target_`:

```yaml
optimizer:
  _target_: src.optimizers.MyAdamW
  lr: 1.0e-3
  weight_decay: 1.0e-2

scheduler:
  _target_: src.schedulers.MyWarmup
  warmup_steps: 2000
```

This works when running `python run.py ...` from the recipe directory (the
recipe root is on `sys.path`, and `src/` is importable as a package. (Ensure
`egs3/<recipe>/<task>/src/__init__.py` exists.)

### Upstream (when reusable across recipes)

If the implementation is broadly reusable, put it under `espnet3/` and use a
fully-qualified `_target_` such as:

```yaml
optimizer:
  _target_: espnet3.components.optimizer.my_optim.MyOptim
```

## 2) How optimizer/schedulers are instantiated

`ESPnetLightningModule.configure_optimizers()` supports exactly two modes:

1. `optimizer` + `scheduler`
2. `optimizers` + `schedulers`

In both modes, the optimizer and scheduler must be provided together (e.g., you
cannot set only `optimizer` without `scheduler`).

Internally it uses Hydra `instantiate(...)` on the config dictionaries:

- Optimizer instantiation is called with a parameter list:
  - single: `instantiate(cfg.optimizer, params)`
  - multiple: `instantiate(cfg.optimizers[i].optimizer, selected_params)`
- Scheduler instantiation is called with the optimizer:
  - single: `instantiate(cfg.scheduler, optimizer=optimizer)`
  - multiple: `instantiate(cfg.schedulers[i].scheduler, optimizer=optimizers[i])`

So your custom classes should follow these conventions:

- **Custom optimizer**: `__init__(self, params, ...)` like `torch.optimizer.Optimizer`
- **Custom scheduler**: `__init__(self, optimizer, ...)` (keyword `optimizer=...`
  is used)

## 3) Single optimizer + scheduler (baseline)

Use when all trainable parameters share one optimizer and scheduler:

```yaml
optimizer:
  _target_: torch.optimizer.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optimizer.lr_scheduler.CosineAnnealingLR
  T_max: 100000
```

### Minimal wiring (what happens in code)

In this mode, `ESPnetLightningModule.configure_optimizers()` does the equivalent of:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 1) module parameters -> optimizer
params = filter(lambda p: p.requires_grad, self.parameters())
optimizer = instantiate(
    OmegaConf.to_container(self.config.optimizer, resolve=True),
    params,
)

# 2) optimizer -> scheduler
scheduler = instantiate(
    OmegaConf.to_container(self.config.scheduler, resolve=True),
    optimizer=optimizer,
)
```

- Do not define `optimizers`/`schedulers` in this mode.
- The scheduler receives the instantiated optimizer automatically.

### Example: `ReduceLROnPlateau` (monitoring a validation metric)

`ReduceLROnPlateau` needs a metric to monitor. In ESPnet3, set
`val_scheduler_criterion` to the logged metric key (e.g., `valid/loss`). This
switches scheduler stepping to `interval: epoch` and passes `monitor` to
Lightning.

```yaml
optimizer:
  _target_: torch.optimizer.AdamW
  lr: 1.0e-3

scheduler:
  _target_: torch.optimizer.lr_scheduler.ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 2

# Key must match what the LightningModule logs (e.g., "valid/loss")
val_scheduler_criterion: valid/loss
```

## 4) Multiple optimizers + schedulers

Enable when different parameter groups need distinct optimizers or learning
rate schedules (e.g., encoder vs. decoder, GAN-style training).

```yaml
optimizers:
  - params: encoder         # substring match against parameter names
    optimizer:
      _target_: torch.optimizer.Adam
      lr: 5.0e-4
      weight_decay: 1.0e-2
  - params: decoder
    optimizer:
      _target_: torch.optimizer.Adam
      lr: 1.0e-3

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

### Minimal wiring (what happens in code)

In this mode, `ESPnetLightningModule.configure_optimizers()` does the equivalent of:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

assert len(self.config.optimizers) == len(self.config.schedulers)

optimizers = []
trainable_params = {
    name: param
    for name, param in self.named_parameters()
    if param.requires_grad
}
used_param_ids = set()

for optim_cfg in self.config.optimizers:
    assert "params" in optim_cfg
    assert "optimizer" in optim_cfg

    selected = [
        p for name, p in trainable_params.items() if optim_cfg["params"] in name
    ]
    selected_names = [
        name for name in trainable_params.keys() if optim_cfg["params"] in name
    ]
    assert len(selected) > 0

    for n in selected_names:
        assert n not in used_param_ids
        used_param_ids.add(n)

    optimizer = instantiate(
        OmegaConf.to_container(optim_cfg["optimizer"], resolve=True),
        selected,
    )
    optimizers.append(optimizer)

unused_param_ids = set(trainable_params.keys()) - used_param_ids
assert not unused_param_ids

optimizer = MultipleOptim(optimizers)

schedulers = []
for i_sch, scheduler in enumerate(self.config.schedulers):
    schedulers.append(
        instantiate(
            OmegaConf.to_container(scheduler.scheduler, resolve=True),
            optimizer=optimizers[i_sch],
        )
    )

scheduler = [
    MultipleScheduler(optimizer, sch, i_sch)
    for i_sch, sch in enumerate(schedulers)
]
```

### How parameter grouping works (`params`)

- Each `optimizers` entry must include `params` and nested `optimizer`.
- `params` is a **substring match** against `ESPnetLightningModule.named_parameters()`.
  In most recipes the actual trainable module is attached as `self.model`, so
  names often look like `model.encoder.layer1.weight` (note the `model.` prefix).
- Only parameters with `requires_grad=True` participate.
- Every trainable parameter must match **exactly one** optimizer.
  - missing coverage → error
  - overlapping substrings → error
- `len(schedulers) == len(optimizers)` is required; they are paired by list position.

### How it runs in Lightning

To keep Lightning in "single optimizer" mode (i.e., not calling
`training_step` once per optimizer), ESPnet3 wraps objects as:

- `optimizer = MultipleOptim([opt1, opt2, ...])`
- `scheduler = [MultipleScheduler(optimizer, sch_i, i), ...]`

Lightning still steps the underlying optimizers/schedulers, but sees them as a
single optimizer interface.

By default, scheduler stepping is configured with `interval: step`. If you set
`val_scheduler_criterion: valid/loss` (or another metric key), ESPnet3 switches
to `interval: epoch` and passes `monitor: <that key>` to Lightning (needed for
`ReduceLROnPlateau`).

### Example: `ReduceLROnPlateau` in multi-optimizer mode

If you use `ReduceLROnPlateau` for any entry under `schedulers`, you still set
`val_scheduler_criterion` once at the top level:

```yaml
optimizers:
  - params: encoder
    optimizer:
      _target_: torch.optimizer.AdamW
      lr: 5.0e-4
  - params: decoder
    optimizer:
      _target_: torch.optimizer.AdamW
      lr: 1.0e-3

schedulers:
  - scheduler:
      _target_: torch.optimizer.lr_scheduler.ReduceLROnPlateau
      mode: min
      factor: 0.5
      patience: 2
  - scheduler:
      _target_: torch.optimizer.lr_scheduler.StepLR
      step_size: 10
      gamma: 0.1

val_scheduler_criterion: valid/loss
```
