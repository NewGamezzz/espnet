---
title: ESPnet3 Train Dataset
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Train Dataset

This page focuses on how training configs describe datasets today.

The full dataset resolution and builder story is documented in:

- [Dataset references and builders](../../core/components/datasets.md)

## Current pattern

Training configs use `DataOrganizer` plus dataset reference entries.

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

Each item can specify the dataset in three ways.

## 1. Omit `data_src` and use the local recipe dataset

If `data_src` is omitted, ESPnet3 loads:

```text
${recipe_dir}/dataset/__init__.py
```

This is what `mini_an4` and `librispeech_100` do in their training configs.

```yaml
train:
  - data_src_args:
      split: train
valid:
  - data_src_args:
      split: valid
```

## 2. Use a dataset tag

```yaml
train:
  - data_src: mini_an4/asr
    data_src_args:
      split: train
```

This resolves to:

```text
egs3.mini_an4.asr.dataset
```

## 3. Use an explicit module path

```yaml
train:
  - data_src: egs3.mini_an4.asr.dataset
    data_src_args:
      split: train
```

## What is forwarded to the dataset constructor

Only `data_src_args` is passed to the exported `Dataset` class:

```python
Dataset(**data_src_args)
```

So a config like:

```yaml
- name: train-clean
  data_src: librispeech_100/asr
  data_src_args:
    split: train-clean-100
    recipe_dir: ${recipe_dir}
```

becomes:

```python
Dataset(split="train-clean-100", recipe_dir=recipe_dir)
```

`name` and `transform` are handled by `DataOrganizer`, not by `Dataset`.

## Current recipe examples

### `mini_an4`

`mini_an4` exports:

- `dataset/__init__.py`
- `dataset/dataset.py`
- `dataset/builder.py`

Its local dataset mode is:

```yaml
dataset:
  recipe_dir: ${recipe_dir}
  train:
    - data_src: mini_an4/asr
      data_src_args:
        split: train
  valid:
    - data_src: mini_an4/asr
      data_src_args:
        split: valid
```

### `librispeech_100`

`librispeech_100` uses the same local-mode pattern, but the dataset reads the
raw LibriSpeech directory directly instead of manifest TSVs.

## Train and valid requirements

Current `DataOrganizer` requires:

- both `train` and `valid`, or
- neither

If training is the goal, define both.

## Test entries

`test` entries are optional for the train stage itself, but adding them in the
same config is useful because:

- inference can reuse the same dataset definition
- measurement can reuse test-set names

Each test entry should define `name`, because that becomes the test-set name
used in `inference_dir/<test_name>/`.

## Related pages

- [Dataset references and builders](../../core/components/datasets.md)
- [DataOrganizer](../../core/components/data-organizer.md)
- [Create dataset stage](../create-dataset.md)
