---
title: ESPnet3 Config Overview
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Config Overview

Current recipe config layout is typically:

```text
egs3/<recipe>/<task>/conf/
  training.yaml
  inference.yaml
  metrics.yaml
  publication.yaml
  demo.yaml
```

The matching CLI flags are:

```bash
python run.py \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

## Default config, overrides, and resolve timing

ESPnet3 recipes use the config files under `egs3/TEMPLATE/<task>/conf/` as default values.
A recipe or user config only overrides the default values it wants to change.

In practice, the flow is:

1. Load the default config from `egs3/TEMPLATE/<task>/conf/`
2. Load the recipe or user config
3. Merge them (override with user config)
4. Resolve interpolations once on the merged result

**Resolve timing**: interpolations are resolved once after the default and user configs are merged.

For example, if the TEMPLATE config contains:

```yaml
recipe_dir: .
exp_tag: training
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

and the user config overrides only:

```yaml
exp_tag: branchformer_debug
```

then the resolved result becomes:

```yaml
exp_tag: branchformer_debug
exp_dir: ./exp/branchformer_debug
```

Overriding `exp_tag` is enough because `exp_dir` is resolved from the merged config.

If you omit a config flag, that stage gets no config and `run.py` raises before execution.

Example:

```text
ValueError: Config not provided for stage(s): train. Use the matching --*_config flag.
```

## Resolvers

ESPnet3 provides:

- `load_line`
- `self_name`

`self_name` is used by TEMPLATE configs such as `exp_tag: ${self_name:}`.
See [Resolvers](./resolvers.md) for details and examples.

## Stage to config mapping

| Stage | Config flag | Typical file |
| --- | --- | --- |
| `create_dataset` | `--training_config` | `training.yaml` |
| `collect_stats` | `--training_config` | `training.yaml` |
| `train` | `--training_config` | `training.yaml` |
| `infer` | `--inference_config` | `inference.yaml` |
| `measure` | `--metrics_config` | `metrics.yaml` |
| `pack_model` | `--training_config` and `--publication_config` | `training.yaml`, `publication.yaml` |
| `upload_model` | `--publication_config` | `publication.yaml` |

## What goes in each config

| File | Purpose |
| --- | --- |
| [`training.yaml`](./train_config.md) | dataset creation, training, dataloader, optimizer, trainer |
| [`inference.yaml`](./infer_config.md) | inference model, test datasets, output writing |
| [`metrics.yaml`](./measure_config.md) | measurement and metric definitions |
| [`publication.yaml`](./publish_config.md) | model packing and upload |

### Notes

- [Training configuration](./train_config.md)
- [Inference configuration](./infer_config.md)
- [Measure stage](../stages/measure.md)
- [Publication configuration](./publish_config.md)
- [Resolvers](./resolvers.md)

## Naming propagation

`run.py` applies runtime context propagation before stage execution:

- training can define canonical `exp_tag` / `exp_dir`
- inference can inherit that identity
- metrics can inherit both experiment identity and `inference_dir`

Example 1: inference inherits training naming.

`training.yaml`:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

`inference.yaml`:

```yaml
exp_tag: inference_beam5
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

Run:

```bash
python run.py \
  --stages train infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```

In this case, inference inherits `training_branchformer`, so the output goes
under:

```text
exp/training_branchformer/inference/
```

Example 2: inference keeps its own naming.

Use the same `inference.yaml`, but run inference alone:

```bash
python run.py \
  --stages infer \
  --inference_config conf/inference.yaml
```

In this case, inference does not inherit training context, so the output goes
under:

```text
exp/inference_beam5/inference/
```

## Related pages

- [Training config](./train_config.md)
- [Inference config](./infer_config.md)
- [Measure config](./measure_config.md)
- [Publication config](./publish_config.md)
- [Resolvers](./resolvers.md)
