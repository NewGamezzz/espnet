---
title: ESPnet3 Config Overview
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Config Overview

ESPnet3 uses separate YAML files per stage. Most recipes follow the layout:

```
egs3/<recipe>/<task>/conf/
  train.yaml
  infer.yaml
  metric.yaml
  publish.yaml
  demo.yaml
```

Each file is passed to `run.py` via the matching CLI flag:

```bash
python run.py \
  --train_config conf/train.yaml \
  --infer_config conf/infer.yaml \
  --metric_config conf/metric.yaml \
  --publish_config conf/publish.yaml \
  --demo_config conf/demo.yaml
```

If you omit a config flag, no config is passed for that stage by default. When a
stage runs without its required config entries, ESPnet3 raises an error so you
can fix the missing settings explicitly.

Example error:

```
ValueError: Config not provided for stage(s): train. Use --train_config/--infer_config/--measure_config.
```

## Resolvers

ESPnet3 registers custom OmegaConf resolvers for loading external files from
YAML. For example, in ASR or speech-to-text recipes you can keep a token list
in a separate text file (to avoid huge YAML blocks) and load it at config time.
See [Resolvers](./resolvers.md) for details.

## Stage to config mapping

| Stage | Config flag | Typical file |
| --- | --- | --- |
| create_dataset | `--train_config` | `train.yaml` |
| collect_stats | `--train_config` | `train.yaml` |
| train | `--train_config` | `train.yaml` |
| infer | `--infer_config` | `infer.yaml` |
| metric | `--metric_config` | `metric.yaml` |
| pack_model | `--train_config` | `train.yaml` |
| upload_model | `--publish_config` | `publish.yaml` |
| pack_demo | `--demo_config` | `demo.yaml` |
| upload_demo | `--demo_config` | `demo.yaml` |

## What goes in each config

| File | Purpose | Typical contents |
| --- | --- | --- |
| [`train.yaml`](./train_config.md) | Training pipeline | model, trainer, optimizers, dataloader, exp_dir |
| [`infer.yaml`](./infer_config.md) | Inference/decoding | model entrypoint, dataset, inference_dir, output_fn, parallel |
| [`metric.yaml`](./metric_config.md) | Scoring/metrics | metrics, inference_dir, test sets |
| [`publish.yaml`](./publish_config.md) | Packaging/upload | pack settings, artifacts to include, HF upload options |
| [`demo.yaml`](./demo_config.md) | Demo build | UI spec, output mapping, infer config path, assets |

### Notes

- Train stage: [Train configuration](./train_config.md)
- Inference config: [Inference configuration](./infer_config.md)
- Metrics pipeline: [Metrics](../stages/metrics.md)
- Demo customization: [Demo guide](../stages/demo.md)
- Resolvers: [Resolvers](./resolvers.md)
