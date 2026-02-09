---
title: ESPnet3 - Callback Mechanisms in Training
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🎛 ESPnet3: Callback Mechanisms in Training

ESPnet3 relies on PyTorch Lightning for training orchestration, so the vast
majority of Lightning callbacks are immediately available.  On top of that, the
`espnet3.components.callbacks` module ships a curated stack of defaults that are
applied when a recipe calls `get_default_callbacks`.  This document explains the
default behaviour and how to extend it in your own experiments.

### Default callback stack

`get_default_callbacks` returns the following callbacks, all pre-configured to
write into the experiment directory:

| Callback | Purpose |
| --- | --- |
| `ModelCheckpoint` (last) | Saves the latest checkpoint (resume entrypoint). |
| `ModelCheckpoint` (best-k) | Saves top-K checkpoints for each metric in `best_model_criterion`. |
| `AverageCheckpointsCallback` | Averages the selected top-K checkpoints and saves a `.pth` file. |
| `LearningRateMonitor` | Logs learning rate values. |
| `TQDMProgressBar` | Shows the progress bar during training. |

1. **`ModelCheckpoint` (last)** – keeps the latest checkpoint under
   `${expdir}` so that training can be resumed.
2. **`ModelCheckpoint` (best-k)** – one instance per entry in
   `best_model_criterion`, each tracking a validation metric and keeping the
   top-*k* checkpoints.
3. **`AverageCheckpointsCallback`** – averages the best checkpoints tracked by
   the callbacks above and writes `&lt;metric&gt;.ave_&lt;k&gt;best.pth` once validation
   finishes.
4. **`LearningRateMonitor`** – logs optimiser learning rates so that they are
   visible in TensorBoard or any Lightning-compatible logger.
5. **`TQDMProgressBar`** – provides an interactive progress bar whose refresh
   rate can be controlled from the configuration file.

All of these callbacks live in
[`espnet3/components/callbacks.py`](../../espnet3/components/callbacks.py) and are
instantiated automatically unless you override the callback list explicitly.

### Controlling checkpoint selection

The metrics that drive checkpoint selection are configured through
`best_model_criterion` in the experiment YAML:

```yaml
best_model_criterion:
  - - valid/loss
    - 3
    - min
  - - valid/wer
    - 2
    - min
```

Each item is interpreted as `(monitor, top_k, mode)`. In this example, ESPnet3
tracks best checkpoints for `valid/loss` and `valid/wer` and (optionally) averages
them via `AverageCheckpointsCallback`.

```yaml
best_model_criterion:
  - - your_metrics
    - 3     # number of ckpts
    - min   # keep the checkpoint if the value is minimal
```

### Example output layout

When checkpoint averaging is enabled, typical outputs under `exp_dir` look like:

```text
${exp_dir}/
  last.ckpt
  epoch0_step1_valid.loss.ckpt
  valid.loss.ave_3best.pth
```

### Adjusting progress logging

The `TQDMProgressBar` refresh interval defaults to 500 steps.  Override the
value by passing `log_interval` when calling `get_default_callbacks` from your
recipe:

```python
from espnet3.components.callbacks import get_default_callbacks

callbacks = get_default_callbacks(
    expdir=str(expdir),
    log_interval=50,
    best_model_criterion=[("valid/loss", 5, "min")],
)
```

Because callbacks are regular Python objects, you can append or replace entries
before constructing the Lightning trainer.

### Providing custom callbacks through Hydra

Recipes can also instantiate callbacks directly from the YAML configuration via
Hydra/OmegaConf.  Simply disable the default stack and enumerate your desired
callbacks in `trainer.callbacks`:

```yaml
trainer:
  callbacks:
    - _target_: espnet3.components.callbacks.default_callbacks.AverageCheckpointsCallback
      output_dir: ${expdir}
      best_ckpt_callbacks:
        - _target_: lightning.pytorch.callbacks.ModelCheckpoint
          monitor: valid/cer
          save_top_k: 5
          mode: min
```

Mixing both approaches is perfectly valid; use `get_default_callbacks` for the
common utilities and append any domain-specific callbacks that your project
requires.

### `AverageCheckpointsCallback` (checkpoint averaging)

This callback loads the top-K checkpoints selected by `best_model_criterion`,
averages model weights, and writes a single `.pth` file.

Key behaviors:

- Output filename: `<metric>.ave_<K>best.pth`
- Averages only parameters under `model.*`
- Runs only on global rank 0
- Uses the checkpoint set produced by the corresponding `ModelCheckpoint` callback(s)

You can reuse the callback outside the defaults by instantiating it directly:

```python
from espnet3.components.callbacks import AverageCheckpointsCallback

ave_ckpt = AverageCheckpointsCallback(
    output_dir=str(expdir),
    best_ckpt_callbacks=[valid_loss_ckpt, valid_wer_ckpt],
)
```

This mirrors the behaviour of `get_default_callbacks` while leaving room for
experiments that require a custom checkpoint policy.
