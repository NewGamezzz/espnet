---
title: ESPnet3 Naming Convention
author:
  name: "Masao Someki"
date: 2026-02-02
---

# ESPnet3 Naming Convention

This page documents naming conventions for ESPnet3 code and docs.

## Use singular vs plural consistently

Use a name that reflects the contents:

- **Plural** when the directory is a collection of similar items.
  - Examples: `components/`, `callbacks/`, `metrics/`, `systems/`, `trainers/`
- **Singular** when the directory is a single concept or a pipeline.
  - Examples: `modeling/`, `demo/`, `parallel/`

## Parts of speech

- **Nouns** for directories, files, classes, and modules.
- **Verbs** for functions and methods.

## Stage names and stage terms

- Stage names are verbs.
- Use the canonical stage names as CLI/config identifiers:
  `create_dataset`, `collect_stats`, `train`, `infer`, `metric`, `publish`, `demo`.
- In ESPnet3, generating hypotheses on test data is **inference** (`infer`).
  Computing metrics from hypotheses is **metric** (`metric`).
- Keep these terms consistent when adding new Systems or stages.
