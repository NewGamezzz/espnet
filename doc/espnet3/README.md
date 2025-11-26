## ESPnet3 Documentation Hub

Start here to navigate the ESPnet3 docs. Pick the path that matches what you
want to do and follow the linked guides.

---

### If you are new to ESPnet3
- **Architecture overview:** [provider_runner.md](./provider_runner.md) explains the Provider/Runner split and how local/cluster execution works.
- **Configurations:** [parallel.md](./parallel.md) shows how to structure Hydra configs and choose a parallel backend.
- **Data flow:** [dataset.md](./dataset.md) describes how datasets, transforms, and preprocessors are wired together.

---

### Preparing data
- **Provider/Runner examples:** [data_preparation.md](./data_preparation.md) walks through building providers and runners for feature extraction and cleaning.
- **Sharding and transforms:** [dataset.md](./dataset.md) covers `CombinedDataset`, `DatasetWithTransform`, and `ShardedDataset`.

---

### Training models
- **Lightning callbacks:** [callbacks.md](./callbacks.md) details the default callback stack and how to customize it.
- **Optimisers and schedulers:** [optimizer_configuration.md](./optimizer_configuration.md) documents single vs. multi-optimiser setups.
- **Multiple optimizers/schedulers:** [multiple_optimizers_schedulers.md](./multiple_optimizers_schedulers.md) shows how to split parameter groups safely.
- **Multi-GPU / multi-node:** [multiple_gpu.md](./multiple_gpu.md) explains training across GPUs or nodes using Lightning configs.

---

### Inference and evaluation
- **Runner-based decoding:** [provider_runner.md](./provider_runner.md) shows how the same runner scales from laptops to clusters.
- **Decode + scoring pipeline:** [evaluate.md](./evaluate.md) outlines `InferenceRunner`, `ScoreRunner`, and YAML-driven metrics.

---

### Tips for navigating recipes
- **Recipe layout:** [recipe_directory.md](./recipe_directory.md) explains the `egs3/` structure, configs, and stage runner.
- **System entry point:** [system.md](./system.md) describes `BaseSystem`/task-specific systems and how stages are executed.
- Keep configs modular (model, trainer, parallel, dataloader blocks).
- Decide on the execution mode early (local vs. SLURM) and set `parallel` accordingly.
- Reuse ESPnet2 model configs where possible; Hydra instantiation keeps both built-in and custom components uniform.
