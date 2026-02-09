---
title: Demo Runtime
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🧠 Demo runtime and inference call path

This page explains how the demo runtime loads the model, executes a single
inference call, and maps the result back to UI outputs.

## 🧭 High-level flow

1. Load `infer.yaml` referenced by `demo.yaml` (resolved relative to the packed demo dir).
2. Resolve provider/runner classes (see `espnet3/demo/resolve.py`).
3. Build the model once via `provider_cls.build_model(infer_cfg)`.
4. On each button click:
   - the UI callback invokes `espnet3/demo/runtime.py:run_inference`
   - collect UI inputs into a dict
   - wrap it in a single-item dataset
   - call `runner_cls.forward(0, dataset=..., model=..., **extras)`
   - map the returned result to UI outputs using `output_keys`

### 🧱 SingleItemDataset

The demo runtime uses a minimal dataset wrapper so it can reuse the standard
runner signature `forward(idx, dataset=..., model=...)` without special-casing
"UI mode".

Flow (per click) in `espnet3/demo/runtime.py:run_inference`:

1. Build a `{name: value}` dict from UI fields.
2. Normalize some inputs.
   - Gradio Audio returns `(sample_rate, np.ndarray)`
   - runtime converts it to a `float32` waveform array via
     `espnet3/demo/runtime.py:_normalize_input`
3. Wrap the dict as a dataset: `dataset = SingleItemDataset(item)`

Implementation (from `espnet3/demo/runtime.py:SingleItemDataset`):

```python
class SingleItemDataset:
    def __init__(self, item):
        self._item = item

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self._item
```

What this means:

- UI inputs are packaged into a single dict like `{"speech": ..., "lang": ...}`.
- `dataset[0]` returns that dict; any other index errors.
- Your runner should read the UI values from `dataset[idx]` (with `idx=0`).
So your runner can treat `dataset[0]["speech"]` (or similar) as a plain waveform
array.
