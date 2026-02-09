---
title: ESPnet3 Systems
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Systems (`espnet3/systems`)

ESPnet3 training is driven by **System** classes. A System is the entry point
that binds configs to stage functions (create_dataset, collect_stats, train,
infer, metric, pack_model, upload_model, pack_demo, upload_demo).
This document explains the base interface, how stages are invoked, and how to
extend it for custom tasks.

## 🧠 BaseSystem at a glance

`espnet3.systems.base.system.BaseSystem` implements the common stage hooks:

```python
class BaseSystem:
    def __init__(
        self,
        *,
        train_config=None,
        infer_config=None,
        metric_config=None,
        publish_config=None,
        demo_config=None,
    ):
        self.train_config = train_config
        self.infer_config = infer_config
        self.metric_config = metric_config
        self.publish_config = publish_config
        self.demo_config = demo_config

    def create_dataset(self): ...
    def collect_stats(self): ...
    def train(self): ...
    def infer(self): ...
    def metric(self): ...

    def pack_model(self): ...
    def upload_model(self): ...
    def pack_demo(self): ...
    def upload_demo(self): ...
```

- Stage methods reject arbitrary CLI arguments; keep settings in YAML configs.
- Defaults delegate to stage functions in `espnet3.systems.base.{train,inference,metric}`.


## How stages are executed

Stage execution in ESPnet3 is **recipe-driven**: a recipe `run.py` constructs a
System instance and then executes stage names by calling
`espnet3.utils.stages_utils.run_stages()`.

The key mechanism is that `run_stages()` resolves each stage name into a method
call on the System instance (via `getattr`) and then invokes it.

Minimal sketch:

```python
from espnet3.utils.stages_utils import run_stages

system = system_cls(...)  # e.g., ASRSystem(...)
run_stages(system=system, stages_to_run=["train", "infer", "metric"])
```

Inside `run_stages()`, the core logic is essentially:

```python
for stage in stages_to_run:
    fn = getattr(system, stage, None)
    fn()  # called with no extra CLI arguments
```

**Note:** A `stage` is simply a method exposed by the System object.
If you add a new method (e.g., `export()`), and include `"export"` in your recipe's stage list, then `run_stages()` will call `system.export()`.

## Updating or adding stages

Stage sets are defined and enforced by each recipe's `run.py` (stage list,
required config guardrails), while the System class provides the actual stage
methods.

For a step-by-step guide to adding a new stage (and optionally a new config),
see: [System-specific Stages](../stages/system-specific.md).

## Directory layout (where things live)

When creating a new System or debugging the stage pipeline, it helps to know
where each "layer" lives:

```text
espnet3/
  systems/
    base/
      system.py        # BaseSystem: stage methods + common wiring
      train.py         # collect_stats(cfg), train(cfg)
      inference.py     # inference(cfg): writes .scp outputs under inference_dir
      metric.py       # metric(cfg): writes metrics.json under inference_dir
    <system>/
      system.py        # Task System (e.g., ASRSystem) overrides/extra stages
      models/          # (Optional) system-owned model definitions (reusable across recipes)
      metrics/         # (Optional) system-owned metric implementations for the metric stage
  components/          # Reusable building blocks (data/training/modeling/metrics/callbacks)
  parallel/            # Provider/Runner runtime (EnvironmentProvider, BaseRunner, backends)
  demo/                # Demo packing/runtime helpers (pack_demo, upload_demo, UI)
```

Recipe structure (what lives under `egs3/`) is documented separately; see
[Recipe directory layout](../recipe_directory.md).

## Creating a new System (practical checklist)

This section focuses on authoring a **System implementation** under
`espnet3/systems/` (not recipe glue code).

### 1) Implement a System class

Create `espnet3/systems/<your_task>/system.py` and subclass `BaseSystem`.
Start by overriding only what your task needs on top of the base pipeline.

```python
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    # Example: task-specific training wrapper
    def train(self):
        # e.g., prepare extra artifacts, choose a task-specific entrypoint,
        # or post-process checkpoints after base training.
        ...

    # Example: task-specific packaging
    def pack_model(self):
        # e.g., select additional artifacts, export formats, customize README/meta.
        ...
```

### 2) Decide what you override vs. what stays in base entrypoints

Typical System-specific customization points:

- `train()` / `collect_stats()`: add pre/post hooks around the training pipeline
- `infer()` / `metric()`: customize inference outputs or evaluation behavior
- `pack_model()` / `upload_model()`: include extra artifacts, export formats, customize README/meta
- `pack_demo()` / `upload_demo()`: generate a task-specific demo bundle
- Extra stages: add new System methods and expose them in the recipe stage list

If you add extra stages, follow: [System-specific Stages](../stages/system-specific.md).

### 3) Keep recipe-specific behavior out of the System

As a rule of thumb, keep the boundary clean:

- **System (`espnet3/systems/...`)**: task-level policy + orchestration (how stages behave)
- **Recipe (`egs3/...`)**: dataset preparation, concrete configs, and experiment assets

If your task needs reusable, system-owned implementations (shared across many
recipes), put them inside the system package:

```text
espnet3/systems/<system>/
  models/   # system-owned models
  metrics/  # system-owned metrics for the metric stage
```

Then reference them from YAML via `_target_` like any other importable class:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.<system>.metrics.my_metric.MyMetric
```

## Add a matching recipe template (TEMPLATE)

When you introduce a new System, also add a matching recipe template under
`egs3/TEMPLATE/<system>/` so new recipes can start from a known-good runner and
default stage list.

At minimum, mirror what `egs3/TEMPLATE/asr/` provides:

```text
egs3/TEMPLATE/<system>/
  run.py   # stage list + CLI entrypoint that calls run_stages()
  conf/    # baseline configs (train/infer/metric/publish/demo as needed)
  src/     # recipe-local code placeholders (models, output_fn, etc.)
```

If you plan to create a PR that adds or changes a System/template, also update the
integration tests so the template stays runnable:

- `ci/test_integration_espnet3.sh` (template-based integration)
- `ci/test_integration_demo.sh` (demo integration)

In particular, keep the MiniAN4-based pipeline working (used by the CI tests)
and extend the tests when your new stages/configs need coverage.
