---
title: ESPnet3 - Data Preparation with Providers and Runners
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🧪 ESPnet3: Data Preparation with Providers and Runners

ESPnet3 replaces the shell-script based pipelines from ESPnet2 with a unified
Python workflow.  Datasets, feature extraction, and data cleaning are performed
through the **Provider/Runner** abstraction that powers both training and
inference.  The same code runs locally on a laptop, on multiprocessing setups,
and on large clusters managed by Dask.

This guide focuses on:

1. Defining providers and runners for data preparation.
2. Executing those runners locally and on clusters.
3. Incorporating data cleaning in the same pipeline.

### 1. Defining the environment: `EnvironmentProvider`

A provider encapsulates everything that has to be instantiated once per
process: datasets, tokenisers, models, or utility objects.  For data preparation
we usually construct datasets and lightweight helpers.  Providers receive a
Hydra configuration so the implementation stays declarative.

```python
from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider

class MyDatasetProvider(EnvironmentProvider):
    def __init__(self, cfg: DictConfig, *, params: dict | None = None):
        super().__init__(cfg)
        self.dataset_cfg = cfg.dataset
        self.cleaner_cfg = getattr(cfg, "cleaner", None)
        self.params = params or {}

    def build_env_local(self):
        dataset = load_raw_samples(self.dataset_cfg)
        cleaner = None if self.cleaner_cfg is None else build_cleaner(self.cleaner_cfg)
        return {"dataset": dataset, "cleaner": cleaner, **self.params}

    def make_worker_setup_fn(self):
        dataset_cfg = self.dataset_cfg
        cleaner_cfg = self.cleaner_cfg
        params = self.params

        def setup():
            dataset = load_raw_samples(dataset_cfg)
            cleaner = None if cleaner_cfg is None else build_cleaner(cleaner_cfg)
            return {"dataset": dataset, "cleaner": cleaner, **params}

        return setup
```

Anything returned by the provider is injected into the runner as keyword
arguments.  Because providers are serialisable, they work in local, parallel
and asynchronous modes without additional changes.

### 2. Implementing the computation: `BaseRunner`

A runner defines a static `forward` method that processes a single index.  The
example below performs feature extraction and includes an optional cleaning
step, showing how preparation and cleaning share the same infrastructure.

```python
from espnet3.parallel.base_runner import BaseRunner
import librosa

class PrepRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset=None, cleaner=None, output_dir=None):
        sample = dataset[idx]
        audio = sample["audio"]
        sr = sample["sr"]
        stft = librosa.stft(audio)

        if cleaner is not None:
            sample = cleaner(sample)
            if sample is None:  # filtered out
                return None

        save_features(sample["utt_id"], stft, output_dir)
        return {
            "utt_id": sample["utt_id"],
            "stft_shape": stft.shape,
        }
```

Because `forward` is static it is pickle-safe and can be shipped to remote
workers.  Any additional runtime arguments (e.g., `cleaner`, `output_dir`) can
be provided through the provider params or when instantiating the runner.

### Optional: batched feature extraction (`batch_forward`)

If your compute can be batched, implement `batch_forward` and pass
`batch_size=<N>` when creating the runner. `BaseRunner` will chunk indices and
call `batch_forward(indices, **env)` instead of `forward(idx, **env)`.

For STFT-like feature extraction, PyTorch supports batched `torch.stft` as long
as you pad to a common length:

```python
from espnet3.parallel.base_runner import BaseRunner
import torch
import torch.nn.functional as F


class PrepRunner(BaseRunner):
    @classmethod
    def batch_forward(cls, indices, *, dataset=None, cleaner=None, output_dir=None):
        samples = [dataset[i] for i in indices]

        if cleaner is not None:
            cleaned = []
            for s in samples:
                s = cleaner(s)
                if s is not None:
                    cleaned.append(s)
            samples = cleaned
            if not samples:
                return []

        wavs = [torch.as_tensor(s["audio"]).float() for s in samples]
        max_len = max(w.shape[-1] for w in wavs)
        wavs = [F.pad(w, (0, max_len - w.shape[-1])) for w in wavs]
        wav = torch.stack(wavs, dim=0)  # (B, T)

        stft = torch.stft(
            wav,
            n_fft=512,
            hop_length=128,
            return_complex=True,
        )  # (B, F, Frames)

        out = []
        for s, feat in zip(samples, stft):
            save_features(s["utt_id"], feat.cpu(), output_dir)
            out.append({"utt_id": s["utt_id"], "stft_shape": tuple(feat.shape)})
        return out
```

### 3. Running locally

Local execution is the default when no parallel configuration is set.  The
provider builds the environment once and the runner iterates through the
requested indices.

```python
from omegaconf import OmegaConf
from espnet3.parallel.base_runner import BaseRunner

cfg = OmegaConf.load("prep.yaml")
provider = MyDatasetProvider(cfg, params={"output_dir": "exp/data"})
runner = PrepRunner(provider)

indices = list(range(len(provider.build_env_local()["dataset"])))
runner(indices)  # returns a list of metadata dictionaries
```

If you implemented `batch_forward`, enable it like:

```python
runner = PrepRunner(provider, batch_size=8)
runner(indices)
```

### 4. Scaling out with Dask (synchronous mode)

To distribute the same runner across multiple workers, define a `parallel`
section in the YAML configuration and pass it to `set_parallel` before
invocation.  No code changes are required.

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 32GB
    walltime: 04:00:00
    job_extra_directives:
      - "--gres=gpu:1"
```

```python
from espnet3.parallel.parallel import set_parallel

set_parallel(cfg.parallel)
results = runner(range(total_items))  # transparently runs on the cluster
```

Workers call `MyDatasetProvider.make_worker_setup_fn()` to instantiate their own
copy of the dataset and helpers before executing `forward`.

### 5. Asynchronous shards for large jobs

For extremely long jobs you can enable the asynchronous mode introduced in the
runner refactor.  Pass `async_mode=True` to the runner and ESPnet3 will
serialize job specifications, submit them through Dask JobQueue, and optionally
store the per-shard outputs as JSONL files:

```python
async_runner = PrepRunner(
    provider,
    async_mode=True,
    async_specs_dir="./specs",
    async_result_dir="./results",
)
job_meta = async_runner(range(total_items))
```

This mode mirrors the behaviour documented in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353): the
same runner can be executed locally, in synchronous SLURM jobs, or via detached
submissions without modifying the preparation logic.
