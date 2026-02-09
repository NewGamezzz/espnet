---
title: ESPnet3 Components
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Components (`espnet3/components`)

This page is a developer-oriented map of the reusable building blocks under
`espnet3/components/`. Components are instantiated from YAML (Hydra/OmegaConf)
and wired by Systems and stage entrypoints.
In ESPnet3, a "component" is any reusable unit that is shared across systems/stages (data, modeling, training, optimization, metrics)

If you can answer "which config key selects which component?", you can extend
ESPnet3 without changing the stage driver.

## Directory map

The `espnet3/components/` directory is currently organized into:

```text
espnet3/components/
├── callbacks/   # Lightning callbacks and related helpers
├── data/        # datasets, organizers, dataloader builders, stats
├── metrics/     # AbsMetric interfaces and metric implementations
├── modeling/    # model wrappers (LightningModule) and model utilities
├── optimizer/       # optimizer/scheduler wiring (incl. multi-optimizer)
└── trainers/    # trainer wrappers and training loop helpers
```

## What you customize (cheat sheet)

| Goal | You change | Where it is used |
| --- | --- | --- |
| Build train/valid/test splits | `dataset` (usually `DataOrganizer`) | `collect_stats`, `train`, `infer`, `metric` |
| Change batching/sharding/collate | `dataloader` | `collect_stats`, `train` |
| Swap the training wrapper (LightningModule/Trainer) | code: `espnet3/components/modeling/*`, `espnet3/components/trainers/*` | `collect_stats`, `train` |
| Add training behaviors | `trainer.callbacks` | `train` |
| Change optimization | `optimizer`/`scheduler` (or multi-optimizer variants) | `train` |
| Add new evaluation metrics | `metrics.metric` + `metrics.inputs` | `metric` |

## Key interfaces (contracts)

These are the "sharp edges" to document and keep stable. When something feels
confusing in components, it usually comes back to one of these contracts.

### Callbacks (training-time hooks)

Callbacks are Lightning objects that run during training (checkpointing, logging,
averaging, progress bars, etc.). ESPnet3 provides a default stack, and you can
extend or override it via `trainer.callbacks` in `train.yaml`.

- Component doc: [Callbacks](./components/callbacks.md)
- Config doc: [Train configuration](../config/train_config.md)

### DataOrganizer and dataset items

ESPnet3 typically builds datasets through `DataOrganizer`. Each dataset item is
an arbitrary dict-like object produced by your dataset class (wrapped by
`DatasetWithTransform` / `CombinedDataset` when configured).

If you plan to implement a custom organizer, see the "Writing a custom DataOrganizer"
section on that page.

- Component doc: [DataOrganizer and dataset pipeline](./components/data-organizer.md)
- Config docs: [Train configuration](../config/train_config.md), [Inference configuration](../config/infer_config.md), [Metric configuration](../config/metric_config.md), [Config overview](../config/index.md)

### Metrics (`AbsMetric`)

Metrics are implemented as `AbsMetric` subclasses and called by the metric
stage with aligned SCP fields.

- Stage doc: [Metric stage](../stages/metrics.md)
- Component doc: [Metrics](./components/metrics.md)
- Config doc: [Metric configuration](../config/metric_config.md)

### Modeling / training wrappers

If you want to change how training is wired (LightningModule/trainer wrappers),
you will typically modify code under `espnet3/components/modeling/` and
`espnet3/components/trainers/` and keep the YAML-facing interface stable.

- Component docs: [Lightning module](./components/model.md), [Trainer](./components/trainer.md)
- Config doc: [Train configuration](../config/train_config.md)

### Optimizers and schedulers

Optimizer/scheduler wiring is config-driven but has strict consistency rules
(especially for multi-optimizer setups).

- Config doc: [Optimizer configuration](./components/optimizer_configuration.md)
- Component docs: [Multiple optimizers & schedulers](./components/multiple_optimizers_schedulers.md)

## "Source of truth": tests

For developer-facing behavior, the tests are the most precise specification.
If you are unsure what a component accepts or returns, check:

- `test/espnet3/components/*`
- `test/espnet3/systems/base/*` (stage wiring behavior)
