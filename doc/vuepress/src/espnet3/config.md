---
title: ESPnet3 Configuration and Parallel Execution
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🧩 ESPnet3 Configuration and Parallel Execution

ESPnet3 leans on OmegaConf/Hydra for declarative experiment setup and on Dask
for parallel execution. This page shows how to structure configs, select
parallel backends, and connect the pieces to training or runner-based jobs.

### ✅ At a glance: who owns what?

| Section      | You edit in YAML                                       | ESPnet3 / libraries handle                           |
| ------------ | ------------------------------------------------------ | ---------------------------------------------------- |
| `model`      | Choose architecture and hyperparameters               | Instantiating via Hydra and wrapping in `ESPnetLightningModule` |
| `trainer`    | Training strategy, devices, logging, callbacks        | Passing options to `lightning.pytorch.Trainer`       |
| `parallel`   | Dask / cluster settings (env, workers, options)       | Creating clients via `espnet3.parallel.parallel.set_parallel` |
| `dataloader` | Dataset, sampler, collate_fn configs                  | Constructing dataloaders and iterating during training |

### Core config layout

Organise YAML files into small, purpose-driven sections. A minimal skeleton
looks like:

```yaml
exp_dir: exp/asr_example
seed: 2025

model:  # ESPnet or custom model
  _target_: my_pkg.models.ASRModel
  encoder_dim: 512

trainer:  # Lightning Trainer arguments
  accelerator: gpu
  devices: 4
  num_nodes: 1
  strategy: ddp

parallel:  # Dask settings for runners/inference
  env: local
  n_workers: 4

dataloader:  # Optional: overrides for collate/sampler/datasets
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    batch_size: 4
    num_workers: 4
  valid:
    batch_size: 4
    num_workers: 4
```

Hydra instantiates each `_target_` at runtime, so the same pattern works for
ESPnet-provided components or your own classes.

### Parallel execution with Dask

`espnet3.parallel.parallel.set_parallel` reads a `parallel` config and prepares a Dask
client. Define multiple blocks (e.g., `parallel_cpu`, `parallel_gpu`) and pick
one at launch.

**Local machine**

```yaml
parallel:
  env: local
  n_workers: 4
  options:
    threads_per_worker: 2
```

**SLURM cluster (GPU)**

```yaml
parallel_gpu:
  env: slurm
  n_workers: 3
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 04:00:00
    job_extra_directives:
      - "--gres=gpu:1"
```

```python
from espnet3.parallel.parallel import set_parallel

set_parallel(cfg.parallel_gpu)
provider = MyProvider(cfg)
runner = MyRunner(provider)
results = runner(range(num_items)) 
```

Switch to asynchronous submission by constructing the runner with
`async_mode=True`; job specs will be written to disk and submitted via Dask
JobQueue.

### Model definition

Two common patterns:

1. **Reuse ESPnet models**
   ```python
   from espnet3.components.modeling.lightning_module import ESPnetLightningModule
   from espnet3.utils.task_utils import get_espnet_model

   espnet_model = get_espnet_model(task="asr", config=cfg.model)
   model = ESPnetLightningModule(espnet_model, cfg)
   ```

2. **Instantiate custom models**
   ```python
   import hydra
   from espnet3.components.modeling.lightning_module import ESPnetLightningModule

   custom_model = hydra.utils.instantiate(cfg.model)
   model = ESPnetLightningModule(custom_model, cfg)
   ```

Both feed directly into the Lightning trainer specified by `trainer`.

### Optimisers and schedulers

ESPnet3 supports single or multiple optimiser setups. See
[Optimizer configuration](./core/components/optimizer_configuration.md) for the
rules enforced by `ESPnetLightningModule.configure_optimizers` (matching
parameter groups, scheduler counts, etc.).

### Dataloader configuration

Attach dataloader settings under `dataloader` to control collate functions,
samplers, and dataset instantiation via Hydra. Example with Lhotse:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    dataset:
      _target_: lhotse.dataset.speech_recognition.K2SpeechRecognitionDataset
      input_strategy:
        _target_: lhotse.dataset.OnTheFlyFeatures
        extractor:
          _target_: lhotse.Fbank
    sampler:
      _target_: lhotse.dataset.sampling.SimpleCutSampler
      max_cuts: 20
      shuffle: true
  valid:
    dataset:
      _target_: lhotse.dataset.speech_recognition.K2SpeechRecognitionDataset
      input_strategy:
        _target_: lhotse.dataset.OnTheFlyFeatures
        extractor:
          _target_: lhotse.Fbank
    sampler:
      _target_: lhotse.dataset.sampling.SimpleCutSampler
      max_cuts: 20
  shuffle: false
```

### Trainer parameters

Most fields under `trainer` are passed straight to `lightning.pytorch.Trainer`.
Objects that need construction (loggers, callbacks, profilers) should be listed
with `_target_` entries so ESPnet3 can instantiate them before handing them to
Lightning.

```yaml
trainer:
  accelerator: gpu
  devices: 4
  num_nodes: 1
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  log_every_n_steps: 500
  max_epochs: 70

  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger

  callbacks:
    # This AverageCheckpointsCallback is included as a default callback without writing here.
    # We included this as an example.
    - _target_: espnet3.components.callbacks.default_callbacks.AverageCheckpointsCallback
      output_dir: ${exp_dir}
      best_ckpt_callbacks: []
```

With this structure, ESPnet3 configurations stay modular while scaling from
single-GPU experiments to large cluster runs with minimal changes.
