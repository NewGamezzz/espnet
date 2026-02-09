---
title: ESPnet3 Parallel
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ⚡ ESPnet3 Parallel (`espnet3/parallel`) — developer landing

This section documents the developer-facing parallel execution utilities used
by ESPnet3 stages and custom recipes. The goal is to let you write computation
once and run it in:

- local (single process)
- synchronous Dask (local multiprocessing / SSH / JobQueue / Kubernetes)
- asynchronous "spec-based" Dask submissions (detached jobs)

If you are only a recipe user, start from the stage pages. This page focuses on
**extension points** and **how the plumbing works**.

## 🧠 Core idea: separate env construction from computation

ESPnet3 parallel is built around two concepts:

- **EnvironmentProvider** (`espnet3/parallel/env_provider.py`): builds the
  per-process environment (dataset/model/tokenizer/etc.).
- **BaseRunner** (`espnet3/parallel/base_runner.py`): runs a static
  `forward(idx, **env)` over many indices, switching between local / Dask / async.

This split makes it safe to ship work to workers without pickling large objects
or capturing `self`.

## 🧩 Main APIs

| Name | File | What you implement / use |
| --- | --- | --- |
| `EnvironmentProvider` | `espnet3/parallel/env_provider.py` | `build_env_local()` and `make_worker_setup_fn()` |
| `InferenceProvider` | `espnet3/parallel/inference_provider.py` | Convenience provider for inference-time dataset/model |
| `BaseRunner` | `espnet3/parallel/base_runner.py` | `forward(...)` (optional `batch_forward`) |
| `set_parallel`, `make_client`, `parallel_for/map` | `espnet3/parallel/parallel.py` | Dask cluster selection and task submission |

### Minimal contracts

- Provider must return a plain `dict` of environment entries.
- Runner `forward` must be `@staticmethod` and accept keyword args that match
  the provider's env keys.

See the full examples in: [Provider / Runner](./parallel/provider_runner.md)

## 🏃 Execution modes

### 1) Local (default)

- `BaseRunner` calls `provider.build_env_local()` on the driver
- then loops `forward(idx, **env)` sequentially

### 2) Synchronous Dask (`parallel_map`/`parallel_for`)

- You configure `cfg.parallel` and call `set_parallel(cfg.parallel)`
- `BaseRunner` calls `provider.make_worker_setup_fn()` and registers it as a
  Dask WorkerPlugin
- tasks are submitted with worker-env injection (name-matching) and gathered
  back to the driver

Key implementation details:

- Worker plugin: `DictReturnWorkerPlugin` in `espnet3/parallel/parallel.py`
- Env injection + conflict detection: `wrap_func_with_worker_env(...)`

### 3) Asynchronous Dask ("spec-based")

`BaseRunner(async_mode=True)`:

1. chunks indices into shards
2. writes one JSON spec per shard to `async_specs_dir` (default `./_async_specs`)
3. submits detached jobs that run `python espnet3/parallel/base_runner.py <spec>`
4. each worker writes JSONL results to `async_result_dir` (default `./_async_results`)

This mode is designed for long-running jobs where you do not want the driver to
stay alive while the cluster runs.

## ⚙️ Parallel config surface

Parallel backends are selected by `cfg.parallel.env` and `cfg.parallel.options`.
Implementation: `espnet3/parallel/parallel.py` (`make_client` / `_make_client`).

Supported `env` values include:

- `local` (Dask `LocalCluster`)
- `local_gpu` (`dask_cuda.LocalCUDACluster`, if installed)
- `ssh` (Dask `SSHCluster`)
- JobQueue: `slurm`, `pbs`, `sge`, `lsf`, `htcondor`, ...
- `kube` (Dask Kubernetes, if installed)

Where to set this in a recipe:

- training: `conf/train.yaml` → `parallel:`
- see also: [Train config](../config/train_config.md)

## 🧰 Design notes and pitfalls

- Prefer building heavy objects (GPU model load, dataset indexing) inside providers, so each worker initializes once.
- Keep env entries serializable and avoid capturing large driver state.
- Avoid arg-name collisions: if your `forward(..., foo=...)` passes `foo` and
  the worker env also contains `foo`, ESPnet3 raises a conflict error.

## 📚 Next pages

- [Provider / Runner](./parallel/provider_runner.md) — contracts, examples, execution modes
- [Inference providers](./parallel/inference_provider.md) — example pattern for inference envs
- [Data preparation](./parallel/data_preparation.md) — example pipeline using provider/runner
- [Multi-GPU / multi-node](./parallel/multiple_gpu.md) — example configs for Lightning/DDP + runners
