---
title: ESPnet2 To ESPnet3 Migration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet2 To ESPnet3 Migration

The main migration shift is:

- ESPnet2: shell-driven recipes and stage numbers
- ESPnet3: Python `run.py`, `System` methods, YAML-first configs

## Current ESPnet3 naming

Use:

- `training.yaml`
- `inference.yaml`
- `metrics.yaml`
- `publication.yaml`

and stages:

- `create_dataset`
- `collect_stats`
- `train`
- `infer`
- `measure`
- `pack_model`
- `upload_model`

## Dataset migration

Recipe-local dataset integration should now center on:

- `dataset/__init__.py`
- `dataset/builder.py`
- `dataset/dataset.py`

not on the older `src/create_dataset.py` / `src/dataset.py` framing as the main
integration story.

## Practical migration checklist

1. copy `egs3/TEMPLATE/<system>/`
2. move config naming to `training.yaml`, `inference.yaml`, `metrics.yaml`,
   `publication.yaml`
3. port data preparation into `DatasetBuilder`
4. port dataset loading into `Dataset`
5. update `run.py` to use current config flags
6. update inference and measurement to `infer` + `measure`

## Recommended references

- [Recipe directory layout](../recipe_directory.md)
- [Dataset references and builders](./components/datasets.md)
- [Systems](./systems.md)
