---
title: 📦 ESPnet3 Data Loading System Documentation
author:
  name: "Masao Someki"
date: 2025-11-26
---

This document provides a comprehensive overview of the dataset system used in ESPnet3, specifically covering:

- `DataOrganizer`
- `DatasetConfig`
- `CombinedDataset`
- `DatasetWithTransform`
- `ShardedDataset` (extension point)
- ESPnet-specific preprocessor behavior

---

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
````

---

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

> ✅ But end users do **not need to set this manually**. The system handles it internally.

---

## ⚙️ DatasetConfig

A dataclass representing a single dataset’s configuration.

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

---

## 🔀 CombinedDataset

Combines multiple datasets into a single `__getitem__`-compatible interface.

### Features

* Applies `(transform, preprocessor)` pair per dataset
* Supports ESPnet-style UID processing if applicable
* Ensures consistent sample keys across datasets
* Optional sharding support (if all datasets subclass `ShardedDataset`)

---

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

---

## 🧩 ShardedDataset

An abstract class representing sharding capability for distributed training.

```python
class MyDataset(ShardedDataset):
    def shard(self, idx):
        return Subset(self, some_index_subset)
```

---

## ⚙️ Preprocessor Behavior in ESPnet3

### Auto-Type Detection

ESPnet3 automatically determines how to call the `preprocessor`:

| Type                          | Call Signature              | Use Case                     |
| ----------------------------- | --------------------------- | ---------------------------- |
| Regular callable              | `preprocessor(sample)`      | Custom/simple processing     |
| Instance of `AbsPreprocessor` | `preprocessor(uid, sample)` | ESPnet’s internal processors |

No need for users to explicitly handle this distinction: **it's handled internally**.

### Intended Responsibilities

| Component    | Role                                     | User Expectations                      |
| ------------ | ---------------------------------------- | -------------------------------------- |
| Dataset      | Load raw data only                       | Implement `__getitem__` returning dict |
| Transform    | Lightweight online modifications         | e.g., normalization, text cleaning     |
| Preprocessor | Mostly for ESPnet2's `CommonPreprocessor` | Follows ESPnet2-supported types only    |

> 🔗 The only officially supported preprocessors are those implemented in
> [espnet2/train/preprocessor.py](https://github.com/espnet/espnet/blob/master/espnet2/train/preprocessor.py)

---

## ✅ Summary

* Users only need to implement `Dataset` (data loading) and `Transform` (modification, optional).
* `Preprocessor` support is automatic, with UID handling taken care of internally.