---
title: 📦 ESPnet3 DataOrganizer
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 DataOrganizer

`DataOrganizer` is the config-driven wrapper that turns dataset entries into the
objects consumed by `collect_stats`, `train`, `infer`, and `measure`.

Implementation:

- `espnet3.components.data.data_organizer.DataOrganizer`

Dataset resolution itself is documented in:

- [Dataset references and builders](./datasets.md)

## Role in the pipeline

```text
training.yaml / inference.yaml
  └── dataset:
        _target_: espnet3.components.data.data_organizer.DataOrganizer
        recipe_dir: ${recipe_dir}
        train: ...
        valid: ...
        test: ...
        preprocessor: ...
```

`DataOrganizer` then:

- builds `train` and `valid` as `CombinedDataset`
- builds each named `test` entry as `DatasetWithTransform`
- applies `transform`, then `preprocessor`

## Current dataset entry format

Each dataset entry uses:

- `data_src`
- `data_src_args`
- `transform`
- `name`

Minimal example:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}

  train:
    - data_src: mini_an4/asr
      data_src_args:
        split: train

  valid:
    - data_src: mini_an4/asr
      data_src_args:
        split: valid

  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test
```

## Important behavior

### `data_src_args` only goes to `Dataset(...)`

`DataOrganizer` resolves the dataset module, gets its exported `Dataset` class,
and instantiates:

```python
Dataset(**data_src_args)
```

Top-level organizer keys such as:

- `name`
- `transform`

stay in organizer space and are not forwarded to the dataset constructor.

### `recipe_dir` matters for local datasets

If a dataset entry omits `data_src`, `DataOrganizer` resolves the local recipe
module:

```text
${recipe_dir}/dataset/__init__.py
```

So local recipes should set:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
```

### `train` and `valid` must move together

Current `DataOrganizer` requires:

- both `train` and `valid` present, or
- both omitted

Providing only one side raises an error.

### `test[*].name` becomes the test-set key

Inference and measurement use the `name` field to choose test sets and to build
per-test output directories.

## Transforms and preprocessor

Each dataset entry may define `transform`, and the organizer itself may define a
shared `preprocessor`.

The order is:

1. `transform(sample)`
2. `preprocessor(sample)` or `preprocessor(uid, sample)`

If the preprocessor is an `AbsPreprocessor`, ESPnet3 uses the ESPnet-style
`(uid, sample)` call automatically.

## Example: local dataset module

`mini_an4` uses the local recipe mode:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - data_src_args:
        split: train
  valid:
    - data_src_args:
        split: valid
```

Because `data_src` is omitted, ESPnet3 loads:

```text
egs3/mini_an4/asr/dataset/__init__.py
```

and instantiates the exported `Dataset`.

## Example: tag-based dataset source

```yaml
test:
  - name: test-clean
    data_src: librispeech_100/asr
    data_src_args:
      split: test-clean
      recipe_dir: ${recipe_dir}
```

This resolves to:

```text
egs3.librispeech_100.asr.dataset
```

## Related pages

- [Dataset references and builders](./datasets.md)
- [Create dataset stage](../../stages/create-dataset.md)
- [Training dataset config](../../stages/train/dataset.md)
