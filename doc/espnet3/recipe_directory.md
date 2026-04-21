---
title: ESPnet3 Recipe Directory Layout
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Recipe Directory Layout

Current recipes under `egs3/` typically look like:

```text
egs3/<recipe>/<task>/
  run.py
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
  dataset/
    __init__.py
    builder.py
    dataset.py
  src/
    inference.py
    tokenizer.py
  data/
  exp/
```

## Where to put what

| Location | You put here | Typical contents |
| --- | --- | --- |
| `egs3/<recipe>/<task>/conf/` | YAML configs | training, inference, metrics, publication |
| `egs3/<recipe>/<task>/dataset/` | dataset module | `Dataset`, `DatasetBuilder`, recipe-local dataset exports |
| `egs3/<recipe>/<task>/src/` | custom Python helpers | inference helpers, tokenizer logic, recipe-local system code |
| `egs3/<recipe>/<task>/run.py` | entry script | stage list, config loading, system selection |

## Important directories

- `conf/`: stage configs
- `dataset/`: recipe-local `Dataset` and `DatasetBuilder`
- `src/`: extra recipe code such as output functions or tokenizer helpers
- `data/`: prepared manifests or recipe-local artifacts
- `exp/`: checkpoints and experiment outputs

## `run.py`

`run.py` loads the stage configs, applies naming propagation, instantiates the
system, and runs the selected stages.

Current config flags are:

- `--training_config`
- `--inference_config`
- `--metrics_config`
- `--publication_config`

Typical stage lists include:

- `create_dataset`
- `collect_stats`
- `train`
- `infer`
- `measure`
- `pack_model`
- `upload_model`

## Directory structure

More broadly, recipes under `egs3/` follow a shared pattern:

```text
egs3/
  TEMPLATE/
    <task>/
      run.py
      conf/
  <recipe>/
    <task>/
      run.py
      conf/
      dataset/
      src/
      data/
      exp/
```

`TEMPLATE` provides the default config values and stage runner wiring, while
each concrete recipe overrides only what it needs.

## Creating a new recipe

1. Copy `egs3/TEMPLATE/<task>` into `egs3/<your_recipe>/<task>`.
2. Add configs under `conf/`.
3. Implement recipe-local dataset code under `dataset/` when needed.
4. Add `src/` helpers such as `inference.py` or recipe-local system code only
   when the recipe needs them.
5. Run stages through `python run.py --stages ...`.

## Notes

The main dataset story is now `dataset/`, not `src/dataset.py` as the primary
integration point.
