---
title: ESPnet3 Components
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Components (`espnet3/components`)

This page is a developer-oriented map of the reusable building blocks under
`espnet3/components/`.

Components are instantiated from YAML and wired by Systems and stage
entrypoints.

In ESPnet3, a "component" is any reusable unit shared across systems or stages:

- data
- modeling
- training
- optimization
- metrics
- callbacks

If you can answer "which config key selects which component?", you can extend
ESPnet3 without changing the stage driver.

## Directory map

The `espnet3/components/` directory is currently organized into:

```text
espnet3/components/
├── callbacks/   # Lightning callbacks and related helpers
├── data/        # datasets, organizers, dataloader builders, stats
├── metrics/     # BaseMetric interfaces and metric implementations
├── modeling/    # LightningModule and optimization wiring
└── trainers/    # trainer wrappers and training loop helpers
```

Current optimizer and scheduler wiring lives under `modeling/`, not under a
separate `optimizer/` package.

## What you customize

| Goal | You change | Where it is used |
| --- | --- | --- |
| Build train/valid/test splits | `dataset` (usually `DataOrganizer`) | `collect_stats`, `train`, `infer`, `measure` |
| Change batching, collate, iterator setup | `dataloader` | `collect_stats`, `train` |
| Swap the training wrapper | code under `espnet3/components/modeling/*` and `espnet3/components/trainers/*` | `collect_stats`, `train` |
| Add training behavior | `trainer.callbacks` | `train` |
| Change optimization | `optimizer` / `scheduler` or `optimizers` / `schedulers` | `train` |
| Add new evaluation metrics | `metrics.metric` + `metrics.inputs` | `measure` |

## Key interfaces

These are the main component contracts.

### Callbacks

Callbacks are Lightning objects that run during training.

Typical roles:

- checkpointing
- logging
- progress bars
- metric summaries

ESPnet3 provides a default stack and lets recipes extend or override it through
`trainer.callbacks` in `training.yaml`.

- Component doc: [Callbacks](./components/callbacks.md)
- Config doc: [Training configuration](../config/train_config.md)

### DataOrganizer and dataset items

ESPnet3 usually builds datasets through `DataOrganizer`.
Each dataset item is an arbitrary dict-like object produced by your dataset
class.

If you plan to implement a custom organizer, see the custom organizer section
on that page.

- Component docs:
  - [DataOrganizer](./components/data-organizer.md)
  - [Dataset references and builders](./components/datasets.md)
- Config docs:
  - [Training configuration](../config/train_config.md)
  - [Inference configuration](../config/infer_config.md)
  - [Measure configuration](../config/measure_config.md)
  - [Config overview](../config/index.md)

### Metrics (`BaseMetric`)

Metrics are implemented as `BaseMetric` subclasses and called by the `measure`
stage with path-based inputs.

- Stage doc: [Measure stage](../stages/measure.md)
- Component doc: [Metrics](./components/metrics.md)
- Config doc: [Measure configuration](../config/measure_config.md)

### Modeling and training wrappers

If you want to change how training is wired, you will usually modify code under:

- `espnet3/components/modeling/`
- `espnet3/components/trainers/`

and keep the YAML-facing interface stable.

- Component docs:
  - [Model](./components/model.md)
  - [Trainer](./components/trainer.md)
- Config doc: [Training configuration](../config/train_config.md)

### Optimizers and schedulers

Optimizer and scheduler wiring is config-driven.
This includes:

- the single-optimizer path
- the named multi-optimizer path

The multi-optimizer path has stricter rules, especially for parameter routing,
gradient clipping, and scheduler naming.

- Component docs:
  - [Optimizer configuration](./components/optimizer_configuration.md)
  - [Multiple optimizers and schedulers](./components/multiple_optimizers_schedulers.md)
- Config doc: [Training configuration](../config/train_config.md)

## Useful references

- [Callbacks](./components/callbacks.md)
- [Dataset references and builders](./components/datasets.md)
- [DataOrganizer](./components/data-organizer.md)
- [Metrics](./components/metrics.md)
- [Model](./components/model.md)
- [Optimizer configuration](./components/optimizer_configuration.md)
- [Multiple optimizers and schedulers](./components/multiple_optimizers_schedulers.md)
- [Trainer](./components/trainer.md)

## Source of truth

For developer-facing behavior, tests are the most precise specification.

If a component contract feels unclear, check:

- `test/espnet3/components/*`
- `test/espnet3/systems/base/*`
