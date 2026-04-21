---
title: ESPnet3 Systems
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Systems (`espnet3/systems`)

ESPnet3 training is driven by **System** classes.
A System binds configs to stage methods such as `create_dataset`, `train`,
`infer`, and `measure`.

This page explains the base interface, how stages are invoked, and how to
extend it for custom tasks.

## BaseSystem at a glance

`espnet3.systems.base.system.BaseSystem` is the common stage wrapper.

Current constructor:

```python
class BaseSystem:
    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        stage_log_mapping=None,
    ):
        ...

    def create_dataset(self): ...
    def collect_stats(self): ...
    def train(self): ...
    def infer(self): ...
    def measure(self): ...

    def pack_model(self): ...
    def upload_model(self): ...
    def pack_demo(self): ...
    def upload_demo(self): ...
```

Current base stages:

| Stage | Method | Default target |
| --- | --- | --- |
| create dataset | `create_dataset()` | dataset module `DatasetBuilder` |
| collect stats | `collect_stats()` | `espnet3.systems.base.training.collect_stats` |
| train | `train()` | `espnet3.systems.base.training.train` |
| infer | `infer()` | `espnet3.systems.base.inference.infer` |
| measure | `measure()` | `espnet3.systems.base.metric.measure` |
| pack model | `pack_model()` | `espnet3.utils.publish_utils.pack_model` |
| upload model | `upload_model()` | `espnet3.utils.publish_utils.upload_model` |
| pack demo | `pack_demo()` | `espnet3.demo.pack.pack_demo` |
| upload demo | `upload_demo()` | `espnet3.demo.pack.upload_demo` |

Notes:

- stage methods do not accept free-form CLI arguments
- keep settings in YAML configs

## How stages are executed

Stage execution is recipe-driven.
A recipe `run.py` builds one System instance and passes stage names to
`run_stages()`.

Minimal sketch:

```python
from espnet3.utils.stages_utils import run_stages

system = system_cls(
    training_config=training_config,
    inference_config=inference_config,
    metrics_config=metrics_config,
    publication_config=publication_config,
)
run_stages(system=system, stages_to_run=["train", "infer", "measure"])
```

The core idea is:

```python
for stage in stages_to_run:
    fn = getattr(system, stage)
    fn()
```

A stage is just a method on the System object.

If you add a new method such as `export()`, and the recipe stage list includes
`"export"`, then `run_stages()` calls `system.export()`.

## Constructor surface

Current config slots are:

| Argument | Used by |
| --- | --- |
| `training_config` | `create_dataset`, `collect_stats`, `train`, `pack_model` |
| `inference_config` | `infer` |
| `metrics_config` | `measure` |
| `publication_config` | `pack_model`, `upload_model` |
| `stage_log_mapping` | optional log-directory override |

`BaseSystem` also stores aliases:

- `train_config -> training_config`
- `infer_config -> inference_config`

## Stage log mapping

`BaseSystem` resolves one log directory per stage.

Base defaults:

| Stage | Path reference |
| --- | --- |
| `create_dataset` | `training_config.data_dir` |
| `collect_stats` | `training_config.stats_dir` |
| `train` | `training_config.exp_dir` |
| `infer` | `inference_config.inference_dir` |
| `measure` | `metrics_config.inference_dir` |
| `pack_model` | `training_config.exp_dir` |
| `upload_model` | `training_config.exp_dir` |

Missing values fall back to:

- `training_config.exp_dir` when available
- otherwise `<cwd>/logs`

Example:

```python
system = BaseSystem(
    training_config=train_cfg,
    inference_config=infer_cfg,
    metrics_config=metrics_cfg,
    stage_log_mapping={
        "infer": "training_config.exp_dir",
        "measure": "training_config.exp_dir",
    },
)
```

## Updating or adding stages

Stage lists and guardrails live in each recipe's `run.py`.
The System class provides the actual methods.

If you want a new stage:

1. add a method to the System
2. add the stage name to the recipe stage list
3. add config loading and required-config checks in `run.py` if needed

You do not have to put every override under `espnet3/systems/...`.
Recipe-local overrides can live under:

```text
egs3/<recipe>/<task>/src/system.py
```

This is useful when the behavior is recipe-specific and should not become a
shared system implementation.

Minimal example:

```python
from espnet3.systems.asr.system import ASRSystem


class RecipeSystem(ASRSystem):
    def export_debug(self):
        output_dir = self.exp_dir / "debug_export"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
```

Then `run.py` can use that class and expose the new stage:

```python
from src.system import RecipeSystem

DEFAULT_STAGES = [
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
    "measure",
    "export_debug",
]

main(args=args, system_cls=RecipeSystem)
```

See [System-specific stages](../stages/system-specific.md).

## Directory layout

This section is mainly for contributors who add or modify systems in
`espnet3/` itself.

If you are only writing a recipe-local override under `egs3/.../src/`, you can
skip this section.

When you send a PR that changes ESPnet3 core systems, it helps to know where
each layer lives.

```text
espnet3/
  systems/
    base/
      system.py        # BaseSystem: stage methods + common wiring
      training.py      # collect_stats(cfg), train(cfg)
      inference.py     # infer(cfg): writes SCP/artifact outputs
      metric.py        # measure(cfg): writes metrics.json
    <system>/
      system.py        # task System (e.g., ASRSystem) overrides/extra stages
      models/          # optional system-owned models
      metrics/         # optional system-owned metrics
  components/          # reusable data/modeling/training pieces
  parallel/            # provider/runner runtime
  demo/                # optional demo packing/runtime
```

Recipe structure under `egs3/` is documented separately.
See [Recipe directory layout](../recipe_directory.md).

## Creating a new System

### 1. Add a System class

Create `espnet3/systems/<your_task>/system.py` and subclass `BaseSystem`.

```python
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    def train(self):
        ...

    def pack_model(self):
        ...
```

Override only the stages your task really needs.

### 2. Decide what stays in base entrypoints

Common override points:

- `train()` / `collect_stats()`
- `infer()` / `measure()`
- `pack_model()` / `upload_model()`
- extra custom stages

Keep the boundary clean:

- `espnet3/systems/...`: task-level behavior
- `egs3/...`: recipe configs, datasets, local assets

### 3. Put reusable task code in the system package

If code is shared across many recipes, keep it under the system package.

Example:

```text
espnet3/systems/<system>/
  models/
  metrics/
```

Then reference it from YAML:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.<system>.metrics.my_metric.MyMetric
```

## Add a matching template

If you add a new System, also add a matching recipe template under
`egs3/TEMPLATE/<task>/`.

At minimum:

```text
egs3/TEMPLATE/<task>/
  run.py
  conf/
  src/
```

`run.py` should:

- define the stage list
- load the needed configs
- instantiate the System
- call `run_stages()`

## Related pages

- [System-specific stages](../stages/system-specific.md)
- [Recipe directory layout](../recipe_directory.md)
- [Training stage](../stages/train.md)
- [Inference stage](../stages/inference.md)
- [Measure stage](../stages/measure.md)
