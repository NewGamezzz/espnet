---
title: ESPnet3 Inference Providers
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🔎 ESPnet3 Inference Providers

Inference workloads usually want the same pattern:

- build a read-only dataset of "items to decode"
- load a model once per worker (often onto GPU)
- run `forward(idx, **env)` repeatedly over indices

`InferenceProvider` is a small convenience base class for this pattern. It is
an `EnvironmentProvider` that standardises the environment keys to:

- `dataset`
- `model`

Implementation reference: `espnet3/parallel/inference_provider.py`.

Related:

- [Provider / Runner](./provider_runner.md)
- [Infer config](../../config/infer_config.md)
- [Inference stage](../../stages/inference.md)
- `espnet3/parallel/base_runner.py`

## ✅ What you implement

Subclasses implement two constructors:

- `build_dataset(cfg)` → returns dataset-like object
- `build_model(cfg)` → returns model-like object

`InferenceProvider` then builds an environment dict like:

```python
{
  "dataset": <your dataset>,
  "model": <your model>,
  **params,  # optional extras
}
```


## 🧩 Minimal example

```python
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.inference_provider import InferenceProvider


class MyProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg):
        return load_samples(cfg.dataset)  # user-defined

    @staticmethod
    def build_model(cfg):
        model = load_model(cfg.model)  # user-defined
        return model.eval()


class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset, model, **env):
        x = dataset[idx]
        return model.decode(x, **env)


provider = MyProvider(cfg, params={"beam_size": 8})
runner = MyRunner(provider)
results = runner(range(len(provider.build_env_local()["dataset"])))
```
