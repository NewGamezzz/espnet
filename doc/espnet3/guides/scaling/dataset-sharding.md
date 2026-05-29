---
title: Dataset Sharding
author:
  name: "Masao Someki"
date: 2026-05-29
---

# Dataset Sharding

When training with multiple GPUs, every rank must see a different, non-overlapping
slice of the data each epoch.
ESPnet3 handles this through dataset sharding: a dataset is split into
`total_shards` pieces, and `DataLoaderBuilder` picks one piece per
`(epoch, rank)` pair automatically.

This page covers:

- the shard rotation formula and how to verify your config with the interactive demo
- the `ShardedDataset` interface you implement
- the YAML wiring
- rules for combining multiple sharded datasets

## Interactive demo

The demo below has three sections.

- **Section 01** — interactive visualizer showing which shard each GPU receives
  per epoch. Adjust the sliders to match your training setup and verify the YAML
  config it generates.
- **Section 02** — responsibility split: what you write, what ESPnet3 handles.
- **Section 03** — three code tabs showing a basic dataset, a sharding-enabled
  dataset, and the multiple-dataset case with the constraints that apply.

<ShardingDemo />

## How shard selection works

`DataLoaderBuilder._maybe_shard_dataset()` runs once at the start of each
epoch and selects one shard for the current `(epoch, rank)` pair:

```
shard_idx = (epoch × world_size + rank) % total_shards
```

This formula guarantees:

- no two ranks ever receive the same shard in the same epoch
- over `total_shards / world_size` epochs, every rank sees every shard exactly
  once

The full dataset is therefore seen by the union of all ranks.
No utterance is permanently skipped.

## YAML config

Sharding is activated by setting `total_shards` and `dist_world_size` on the
dataset entry inside the `data` block of `training.yaml`.

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - data_src: egs3.my_recipe.asr.dataset.builder
      data_src_args:
        split: train
        total_shards: 16
        dist_world_size: 16
```

And the corresponding dataloader config:

```yaml
dataloader:
  train:
    total_shards: 16
    dist_world_size: 16
    iter_factory:
      ...
```

Both the dataset and the dataloader must agree on the same values.

### Validation rules

ESPnet3 enforces the following checks at startup and raises a `RuntimeError`
if any condition is violated:

| Condition | Error |
| --- | --- |
| `dist_world_size` ≠ runtime `world_size` | `dist_world_size must match the distributed world_size` |
| `total_shards % world_size ≠ 0` | `total_shards must be divisible by world_size` |
| `total_shards` is unset on a `ShardedDataset` | `total_shards is set but shard() is not implemented` |
| Mix of `ShardedDataset` and plain `Dataset` in one `CombinedDataset` | `If any dataset is a subclass of ShardedDataset, then all dataset should be a subclass of ShardedDataset` |
| Datasets disagree on `total_shards` or `dist_world_size` | `All sharded datasets must share the same total_shards and dist_world_size` |

### Single-GPU runs

Leave both at `1` (the default):

```yaml
dataloader:
  train:
    total_shards: 1
    dist_world_size: 1
```

No sharding is applied.
`DataLoaderBuilder` returns the full dataset as-is.

## Writing a sharded dataset

### The ShardedDataset interface

Subclass `espnet3.components.data.dataset.ShardedDataset` and implement three
things:

1. Set `total_shards` and `dist_world_size` as instance attributes.
2. Implement `__getitem__` and `__len__` as for any PyTorch dataset.
3. Implement `shard(shard_idx)` to return a `Dataset` covering only that shard.

```python
from torch.utils.data import Dataset, Subset

from espnet3.components.data.dataset import ShardedDataset


class MyASRDataset(ShardedDataset):

    def __init__(
        self,
        data_dir: str,
        split: str,
        total_shards: int = 8,
        dist_world_size: int = 4,
    ):
        self.samples = load_manifest(data_dir, split)
        self.total_shards = total_shards
        self.dist_world_size = dist_world_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        return {
            "speech": load_audio(item["path"]),
            "text": item["transcript"],
        }

    def shard(self, shard_idx: int) -> Dataset:
        n = len(self.samples)
        shard_size = n // self.total_shards
        start = shard_idx * shard_size
        return Subset(self, list(range(start, start + shard_size)))
```

### `__len__` semantics

`__len__` on a `ShardedDataset` returns the total number of samples
**across all shards** (the full pre-sharding dataset size).

ESPnet3 never calls `len(dataset)` directly for DataLoader construction.
It calls `len(dataset.shard(shard_idx))` instead.
The `__len__` you define is required only to satisfy PyTorch's `Dataset` ABC.

### `shard()` contract

`shard(shard_idx)` must return any object that implements `__len__` and
`__getitem__`.
`torch.utils.data.Subset` is the most common return type, but a sliced list,
a custom wrapper, or even another `Dataset` subclass are all valid.

You never call `shard()` yourself.
`DataLoaderBuilder` calls it once per epoch with the correct `shard_idx`
for this GPU.

### Passing sharding parameters from YAML

`total_shards` and `dist_world_size` are usually passed through `data_src_args`
in `training.yaml`:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - data_src: egs3.my_recipe.asr.dataset.builder
      data_src_args:
        split: train
        total_shards: ${dataloader.train.total_shards}
        dist_world_size: ${dataloader.train.dist_world_size}
```

Using Hydra interpolation keeps the values in one place.
The dataset receives them as keyword arguments at construction time.

## Multiple datasets in one split

`DataOrganizer` combines multiple datasets into a single `CombinedDataset`
for each split.
When sharding is involved, `CombinedDataset` imposes two additional constraints:

1. **All datasets must be `ShardedDataset` subclasses.**
   Mixing a `ShardedDataset` with a plain `Dataset` in the same split raises
   a `RuntimeError`.

2. **All datasets must agree on `total_shards` and `dist_world_size`.**
   `CombinedDataset` reads these values from every dataset in the list and
   raises a `RuntimeError` if any pair differs.

```yaml
train:
  - data_src: egs3.my_recipe.asr.dataset.builder   # total_shards=8
    data_src_args:
      split: train
      total_shards: 8
      dist_world_size: 4
  - data_src: egs3.my_recipe.asr.dataset.extra      # total_shards=8 ← must match
    data_src_args:
      split: train
      total_shards: 8
      dist_world_size: 4
```

When `CombinedDataset.shard(shard_idx)` is called, it calls
`dataset.shard(shard_idx)` on each component dataset and wraps the results
in a new `CombinedDataset` of the same shape.

### Output key consistency

`CombinedDataset` checks at construction time that every dataset returns
the same set of keys from `__getitem__`.
This check applies with or without sharding.

If two datasets return different keys, `CombinedDataset` raises an
`AssertionError` immediately rather than failing silently during training.

## Choosing total_shards

`total_shards` must be divisible by `dist_world_size`.
Beyond that constraint, a few rules of thumb:

| Situation | Recommendation |
| --- | --- |
| `total_shards == dist_world_size` | Each GPU owns exactly one shard forever — no shard rotation across epochs. Use only when each shard is large enough to train for many steps. |
| `total_shards` is a small multiple of `dist_world_size` | Rotation kicks in over a few epochs. Balanced coverage with moderate shard overhead. |
| `total_shards` is a large multiple of `dist_world_size` | Fine-grained rotation — each GPU sees a different slice every epoch. Useful when the dataset is very large and shard construction is cheap. |

For most recipes, setting `total_shards` to 2–4× `dist_world_size` is a
reasonable default.

## Common mistakes

**`dist_world_size` left at `1` for a multi-GPU run.**
Set `dist_world_size: ${num_nodes * num_device}` or compute the product
explicitly.
The runtime world size is determined by `torch.distributed.get_world_size()`,
not by any ESPnet3 config field.

**`total_shards` not divisible by `dist_world_size`.**
For example, `total_shards: 10` with `dist_world_size: 8` will fail at
DataLoader construction.

**Mixing a `ShardedDataset` and a plain `Dataset` in the same split.**
Both datasets in the same split must subclass `ShardedDataset`.
Move sharding-incompatible datasets to a separate split, or add a trivial
`shard()` implementation that returns `self`.

**Datasets in the same split disagree on `total_shards`.**
This usually happens when two datasets have hard-coded defaults that differ.
Pass both values through `data_src_args` from a shared YAML interpolation
target to keep them in sync.

**Implementing `shard()` to return overlapping indices.**
If two shards share indices, some utterances are seen twice and others never.
Verify shard coverage by checking `sum(len(ds.shard(i)) for i in range(total_shards)) == len(ds)`.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Large-scale data"
    desc="batch_bins, dataloader total_shards and dist_world_size in training.yaml."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
  <DocCard
    title="Multi-node training"
    desc="trainer.num_nodes and how dist_world_size is calculated."
    icon="tabler:topology-star"
    href="./multi-node.html"
  />
  <DocCard
    title="Dataloader Config"
    desc="Full reference for iter_factory, collate_fn, and batch strategies."
    icon="tabler:settings-2"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="Datasets"
    desc="Dataset builders, DataOrganizer, and CombinedDataset internals."
    icon="tabler:layers-intersect"
    href="../../core/components/datasets.html"
  />
</DocCards>
