---
title: Multiple Optimizers and Schedulers in ESPnet3
---

ESPnet3 lets you configure one optimizer/scheduler pair or multiple optimizer
and scheduler groups. This guide explains the rules enforced by
`LitESPnetModel.configure_optimizers`, how parameter selection works, and how
to debug common errors.

## Single optimizer + scheduler (baseline)

Use when all trainable parameters share one optimizer and scheduler:

```yaml
optim:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100000
```

- Do not define `optims`/`schedulers` in this mode.
- The scheduler receives the instantiated optimizer automatically.

## Multiple optimizers + schedulers

Enable when different parameter groups need distinct optimizers or learning
rate schedules (e.g., encoder vs. decoder, GAN-style training).

```yaml
optims:
  - params: encoder         # substring match against parameter names
    optim:
      _target_: torch.optim.Adam
      lr: 5.0e-4
      weight_decay: 1.0e-2
  - params: decoder
    optim:
      _target_: torch.optim.Adam
      lr: 1.0e-3

schedulers:
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 5
      gamma: 0.5
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 10
      gamma: 0.1
```

### How grouping works

- Each `optims` entry must include `params` and `optim`.
- `params` is matched as a substring against **named parameters**. All matches
  go to that optimizer.
- Every trainable parameter must match **exactly one** optimizer. ESPnet3 will
  error if a parameter is missing or matched twice.
- The number of `schedulers` must equal the number of `optims`; they are paired
  by position (first scheduler controls the first optimizer, etc.).
- Internally ESPnet3 wraps the list in `MultipleOptim` and `MultipleScheduler`
  so Lightning still sees a single optimizer with step-level scheduling.

### Common pitfalls and fixes

- **No trainable parameters found for substring**  
  Check `params` strings against `model.named_parameters()`. Consider using a
  more specific prefix (e.g., `encoder.`).

- **Parameter is assigned to multiple optimizers**  
  Ensure substrings do not overlap unintentionally (e.g., `enc` and `encoder`
  both matching). Make substrings mutually exclusive.

- **Unused parameters reported**  
  Add another `optims` entry or adjust substrings so every trainable parameter
  is covered.

- **Mixed modes**  
  Do not mix `optim` with `optims` or `scheduler` with `schedulers` in the same
  config.

### Debugging tip

To see which parameters matched each group, insert a quick check before
training:

```python
for name, _ in model.named_parameters():
    if "encoder" in name:
        print("encoder group:", name)
    if "decoder" in name:
        print("decoder group:", name)
```

Use this to refine the `params` substrings until grouping matches your intent.
