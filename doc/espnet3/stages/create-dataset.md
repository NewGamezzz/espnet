---
title: ESPnet3 Create Dataset Stage
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Create Dataset Stage

## How to run

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

## Where it is configured

`create_dataset` is configured from the `create_dataset` block in
`training.yaml`.

Example:

```yaml
dataset_dir: ${recipe_dir}/data/mini_an4

create_dataset:
  recipe_dir: ${recipe_dir}
  dataset_dir: ${dataset_dir}
```

Those values are forwarded as keyword arguments to the builder methods.

## What the stage does

`create_dataset()` walks over `training_config.dataset.train`,
`training_config.dataset.valid`, and `training_config.dataset.test`.

For each unique dataset source, it:

1. resolves the dataset module
2. instantiates `DatasetBuilder`
3. runs `is_source_prepared(**create_dataset)`
4. runs `prepare_source(**create_dataset)` if needed
5. runs `is_built(**create_dataset)`
6. runs `build(**create_dataset)` if needed

The same dataset source is only prepared once per stage run.

## Builder lifecycle

The current contract is:

- `is_source_prepared(**kwargs) -> bool`
- `prepare_source(**kwargs) -> None`
- `is_built(**kwargs) -> bool`
- `build(**kwargs) -> None`

The intended split is:

- source preparation: download, extract, validate, or locate raw assets
- build: run task-ready preprocessing or other recipe-local dumping data

## Where builder kwargs come from

The arguments passed to the builder methods come from the `create_dataset`
block in `training.yaml`.

Example:

```yaml
create_dataset:
  recipe_dir: ${recipe_dir}
  source_dir: ${dataset_dir}
```

That means the stage calls:

```python
builder.is_source_prepared(recipe_dir=..., source_dir=...)
builder.prepare_source(recipe_dir=..., source_dir=...)
builder.is_built(recipe_dir=..., source_dir=...)
builder.build(recipe_dir=..., source_dir=...)
```

## Where the code lives

Typical recipe structure:

```text
egs3/<recipe>/<task>/
  conf/
    training.yaml
  dataset/
    __init__.py
    builder.py
    dataset.py
```

`builder.py` handles source preparation and build-time side effects.
`dataset.py` defines the runtime `Dataset` class used by train and inference.

## `dataset/__init__.py`

`dataset/__init__.py` should export these two names:

- `Dataset`
- `DatasetBuilder`

`create_dataset` looks up `DatasetBuilder`.
Training and inference look up `Dataset`.

Minimal example:

```python
from egs3.my_recipe.asr.dataset.builder import MyDatasetBuilder as DatasetBuilder
from egs3.my_recipe.asr.dataset.dataset import MyDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

This is the contract for all three resolution modes:

- dataset tag
- explicit module path
- omitted `data_src` -> `${recipe_dir}/dataset/__init__.py`

## How dataset modules are resolved

Dataset resolution is shared with the normal dataset loading path:

1. `data_src: mini_an4/asr`
2. `data_src: egs3.mini_an4.asr.dataset`
3. omitted `data_src`, which loads `${recipe_dir}/dataset/__init__.py`

Details are in:

- [Dataset references and builders](../core/components/datasets.md)

## Example: `mini_an4`

`egs3/mini_an4/asr/dataset/builder.py` is a full build example.

Behavior:

- `prepare_source()` extracts the AN4 archive under the recipe dataset area
- `build()` converts audio and writes manifest TSVs under `data/manifest/`

The resulting tree is roughly:

```text
egs3/mini_an4/asr/
  data/
    manifest/
      train.tsv
      valid.tsv
      test.tsv
    wav/
      train/
      test/
```

Minimal conceptual export:

```python
from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.asr.dataset.dataset import MiniAn4Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

## Example: `librispeech_100`

`egs3/librispeech_100/asr/dataset/builder.py` is the contrasting pattern.

Behavior:

- `prepare_source()` only validates that the LibriSpeech tree exists
- `is_built()` simply reuses source readiness
- `build()` is effectively a no-op validation path

This recipe reads the original corpus layout directly instead of generating
separate manifests.

This is the contrasting pattern to `mini_an4`: the builder still participates
in the stage lifecycle, but the recipe chooses not to materialize a separate
manifest representation.

## Recommended implementation pattern

A new recipe dataset module should export:

```python
from egs3.my_recipe.asr.dataset.builder import MyBuilder as DatasetBuilder
from egs3.my_recipe.asr.dataset.dataset import MyDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

with:

- `DatasetBuilder` for `create_dataset`
- `Dataset` for training / inference dataset instantiation

## Notes

- `create_dataset` should be deterministic and safe to re-run
- source preparation and build are intentionally separate checks
- the same dataset source is only prepared once even if it appears in multiple
  splits

## Related pages

- [Dataset references and builders](../core/components/datasets.md)
- [DataOrganizer](../core/components/data-organizer.md)
- [Training dataset config](./train/dataset.md)
