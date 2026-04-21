---
title: ESPnet3 Inference Provider
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Provider

`InferenceProvider` is the convenience provider base for inference-time
dataset/model construction.

Implementation:

- `espnet3.parallel.inference_provider.InferenceProvider`

## What subclasses implement

- `build_dataset(config)`
- `build_model(config)`

`InferenceProvider` then builds an environment containing:

- `dataset`
- `model`
- any extra `params`

## Notes

- local execution reuses a cached local environment
- worker execution rebuilds the environment through
  `build_worker_setup_fn()`

## Related pages

- [Provider / runner](./provider_runner.md)
- [Inference stage](../../stages/inference.md)
