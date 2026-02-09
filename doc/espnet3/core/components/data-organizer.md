---
title: 📦 ESPnet3 DataOrganizer and Dataset Pipeline
author:
  name: "Masao Someki"
date: 2025-11-26
---

This document provides an overview of the dataset pipeline used in ESPnet3, covering:

- `DataOrganizer`
- `DatasetConfig`
- `CombinedDataset`
- `DatasetWithTransform`
- `ShardedDataset` (extension point)
- ESPnet-specific preprocessor behavior

## 🧠 System Overview

```text
dataset.yaml
   ↓
Hydra (config construction)
   ↓
DataOrganizer
   ├── CombinedDataset (train / valid)
   │     └─ (transform, preprocessor) per dataset
   └── DatasetWithTransform (test)
```

## 🔗 DataOrganizer

### Purpose


`DataOrganizer` constructs and organizes train, validation, and test datasets using Hydra configuration dictionaries. It wraps datasets into unified interfaces for data loading and transformation.

### Behavior

* **Train / Valid** → Combined into `CombinedDataset`
* **Test** → Wrapped in individual `DatasetWithTransform` instances
* Each sample flows through:

  1. `transform(sample)`
  2. `preprocessor(sample)` or `preprocessor(uid, sample)`

### Automatic Preprocessor Handling

In ESPnet3, the type of `preprocessor` is **automatically inferred**:

* If it is an instance of `AbsPreprocessor`, the call will use `(uid, sample)`
* Otherwise, it is called with a single argument: `sample`

This means users **do not need to worry** about manually providing `uid`.

```python
# Internally handled:
sample = transform(raw_sample)
sample = preprocessor(sample)         # or
sample = preprocessor(uid, sample)    # if it's AbsPreprocessor
```

### ESPnet-Specific Note

When training, ESPnet's `CommonCollator` expects `(uid, sample)` pairs. To support this:

```python
organizer.train.use_espnet_collator = True
sample = organizer.train[0]  # Returns (uid, sample)
```

✅ But end users do **not need to set this manually**. The system handles it internally.

## 🧾 Minimal YAML example

`DataOrganizer` is typically configured under the `dataset` section of your
stage YAML (e.g., `train.yaml`, `infer.yaml`, `metric.yaml`).

Minimal example that defines `train`, `valid`, and `test` splits:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer

  train:
    - name: train
      dataset:
        _target_: src.dataset.MyDataset
        split: train
      transform:
        _target_: src.transforms.my_transform

  valid:
    - name: valid
      dataset:
        _target_: src.dataset.MyDataset
        split: valid

  test:
    - name: test
      dataset:
        _target_: src.dataset.MyDataset
        split: test
```

Notes:

- Provide **both** `train` and `valid` (or omit both). Providing only one of
  them is invalid.
- `test[*].name` becomes the key used by inference/metrics to select a
  split (via `config.test_set` / `dataset.test` enumeration).

## ⚙️ DatasetConfig

A dataclass representing a single dataset's configuration.

```yaml
- name: dev-clean
  dataset:
    _target_: my_project.datasets.MyDataset
    split: dev-clean
  transform:
    _target_: my_project.transforms.to_upper
```

### Fields

| Field       | Description                            |
| ----------- | -------------------------------------- |
| `name`      | Dataset name                           |
| `dataset`   | Hydra config for dataset instantiation |
| `transform` | Hydra config for transformation instantiation |

## 🔀 CombinedDataset

Combines multiple datasets into a single `__getitem__`-compatible interface.

### Features

* Applies `(transform, preprocessor)` pair per dataset
* Supports ESPnet-style UID processing if applicable
* Ensures consistent sample keys across datasets
* Optional sharding support (if all datasets subclass `ShardedDataset`)

## 🎛 DatasetWithTransform

A lightweight wrapper for applying a single `(transform → preprocessor)` to a dataset. Used primarily for test sets.

```python
wrapped = DatasetWithTransform(
    dataset,
    transform,
    preprocessor,
    use_espnet_preprocessor=True
)
```

## 🧩 ShardedDataset

An abstract class representing sharding capability for distributed training.

```python
class MyDataset(ShardedDataset):
    def shard(self, idx):
        return Subset(self, some_index_subset)
```

## ⚙️ Preprocessor Behavior in ESPnet3

### Auto-Type Detection

ESPnet3 automatically determines how to call the `preprocessor`:

| Type                          | Call Signature              | Use Case                     |
| ----------------------------- | --------------------------- | ---------------------------- |
| Regular callable              | `preprocessor(sample)`      | Custom/simple processing     |
| Instance of `AbsPreprocessor` | `preprocessor(uid, sample)` | ESPnet's internal processors |


### Intended Responsibilities

| Component    | Role                                     | User Expectations                      |
| ------------ | ---------------------------------------- | -------------------------------------- |
| Dataset      | Load raw data only                       | Implement `__getitem__` returning dict |
| Transform    | Lightweight online modifications         | e.g., normalization, text cleaning     |
| Preprocessor | Mostly for ESPnet2's `CommonPreprocessor` | Follows ESPnet2-supported types only    |

🔗 The only officially supported preprocessors are those implemented in
[espnet2/train/preprocessor.py](https://github.com/espnet/espnet/blob/master/espnet2/train/preprocessor.py)

## 🧩 Writing a custom DataOrganizer

Most users will not need a custom organizer (the built-in `DataOrganizer` is
config-driven), but if you want to implement your own organizer class, the
important part is matching the expectations of the stages that consume it.

### Where it is used (what must work)

Think of this as "what code will run against your organizer".

#### Training / collect_stats


Your organizer is expected to expose `train` and `valid` datasets that behave
like PyTorch datasets:

```python
organizer = instantiate(cfg.dataset)  # _target_: ...DataOrganizer or your custom class

train_ds = organizer.train
valid_ds = organizer.valid

assert len(train_ds) > 0
sample0 = train_ds[0]  # usually a dict, or (uid, dict) in ESPnet-collator mode
```

If you support ESPnet2-style training that expects `(uid, sample)` pairs, you
need a switch that changes the return type:

```python
# Example pattern used by CombinedDataset
train_ds.use_espnet_collator = True
uid, sample = train_ds[0]
assert isinstance(uid, str)
assert isinstance(sample, dict)
```

#### Inference


Inference selects a named test set via `config.test_set` and expects your
organizer to expose a `test` mapping:

```python
organizer = instantiate(cfg.dataset)
test_name = cfg.test_set  # set by the inference loop for each dataset.test entry

test_ds = organizer.test[test_name]  # dict-like lookup
item = test_ds[0]                    # dict-like dataset item
```

If `organizer.test[test_name]` fails, inference cannot route to the requested
split.

### Input/output contracts to keep in mind

#### Dataset item <span class='small-bracket'>(input)</span>


- Dataset items should be **dict-like** (a plain `dict` is ideal).
- Keys must be consistent across datasets when you combine them (the default
  `CombinedDataset` enforces this by checking sample keys).

#### Transform and preprocessor <span class='small-bracket'>(mid-pipeline)</span>


- `transform(sample)` should return a dict.
- `preprocessor(sample)` or `preprocessor(uid, sample)` should return a dict.
- Avoid mutating shared objects in-place unless you know your dataset is not
  reused across workers/processes.

#### Train-time collator mode <span class='small-bracket'>(output)</span>


If you support an ESPnet-style collator mode, the dataset output becomes:

```text
(uid: str, sample: dict)
```

where `uid` is stable and unique within the dataset.

### Minimal skeleton (conceptual)

```python
class MyOrganizer:
    def __init__(self, *, train=None, valid=None, test=None, preprocessor=None):
        self.train = build_train_dataset(train, preprocessor=preprocessor)
        self.valid = build_valid_dataset(valid, preprocessor=preprocessor)
        self._test_sets = build_test_sets(test, preprocessor=preprocessor)

    @property
    def test(self):
        return self._test_sets  # dict-like: name -> dataset
```

### If you plan to upstream this (PR guidance)

If you are planning to submit a PR that changes `DataOrganizer` behavior or adds
a new organizer abstraction, implement it while running the existing unit tests
as a spec:

- `test/espnet3/components/data/test_data_organizer.py`

That test suite is the most concrete description of the current expected
behavior (train/valid consistency, test set mapping, key consistency, and
collator mode expectations).

### Common pitfalls

- **Providing only `train` or only `valid`**: training assumes both exist (or neither).
- **Inconsistent sample keys** across datasets: collate functions and models will break.
- **Missing test name mapping**: inference cannot select a test set if `organizer.test[name]` fails.
- **Unstable or missing `utt_id`**: downstream `.scp` writing and metrics alignment become painful.
