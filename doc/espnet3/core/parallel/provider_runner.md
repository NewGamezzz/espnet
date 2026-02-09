---
title: ESPnet3 - Provider/Runner Architecture
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🏗 ESPnet3: Provider/Runner Architecture

The Provider/Runner split introduced in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3393671961)
separates environment construction from computation.  This section summarises
how to author new providers and runners and how they enable seamless execution
from laptops to clusters.

### Responsibilities

- **EnvironmentProvider** – builds everything that should live on a worker
  (datasets, models, tokenisers, helper objects).  The provider receives the Hydra
  configuration and exposes two methods:
  - `build_env_local()` for pure local runs.
  - `make_worker_setup_fn()` returning a callable that constructs the environment
    on each worker.
- **BaseRunner** – implements a static `forward(idx, **env)` method.  The runner
  is pickle-safe and operates purely on the dictionaries created by the provider.

By restricting state to these two pieces ESPnet3 ensures that the same Python
code works in local, multiprocessing, and Dask JobQueue modes.

### Minimal example

- Inference example

```python
from espnet3.parallel.base_runner import BaseRunner
from espnet3.systems.base.inference_provider import InferenceProvider

class MyProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg):
        return load_samples(cfg.dataset)

    @staticmethod
    def build_model(cfg):
        model = build_model(cfg.model)
        return model.to(cfg.model.device)

class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset, model, **env):
        sample = dataset[idx]
        return model.decode(sample, beam_size=env.get("beam_size", 5))

provider = MyProvider(cfg, params={"beam_size": 8})
runner = MyRunner(provider)
num_items = len(provider.build_env_local()["dataset"])
outputs = runner(range(num_items))  # works locally
```

- Example on your custom provider


```python
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.base_runner import BaseRunner

class MyProvider(EnvironmentProvider):
    @staticmethod
    def build_env_local(cfg):
        return {"abc": 123}

    @staticmethod
    def make_worker_setup_fn(cfg):
        def setup_fn():
            return {"abc": 123}
        return setup_fn

class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, abc, **env):
        with open("abc.txt", "w") as f:
            f.write(f"{idx}: {abc}\n")

provider = MyProvider(cfg)
runner = MyRunner(provider)
outputs = runner(range(3))
```

Switch to distributed execution by calling `set_parallel(cfg.parallel)` or by
constructing the runner with `async_mode=True`; no further changes are required.

### Execution modes

The same runner can be executed in three modes:

1. **Local** – default when no parallel configuration is set.
2. **Synchronous** – when `set_parallel` is called, ESPnet3 uses a shared Dask
   client and returns results to the driver.
3. **Asynchronous** – when `async_mode=True` the runner emits JSON specs,
   replaces the Dask JobQueue submission command, and launches detached jobs.

Real-world metrics for the OWSM-V4 medium (1B) inference runner are
published in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353).  The
results show that scaling from one to four GPUs on SLURM reduces wall time from
1336 s to 369 s without touching the Python code.

### Customising Dask job submissions

In async mode, `BaseRunner` dynamically subclasses the cluster's `job_cls` and
overrides the submission command so the worker runs a spec file.

Implementation pointers:

- `espnet3/parallel/base_runner.py:get_job_cls()` defines `ASyncRunnerJob`
  (a runtime subclass of the underlying JobQueue job class).
- `ASyncRunnerJob` rewrites `self._command_template` to run
  `python espnet3/parallel/base_runner.py <spec_path> ...`.
- `espnet3/parallel/base_runner.py:BaseRunner._run_async()` installs this via:
  `client.cluster.job_cls = get_job_cls(client.cluster, spec_path)`.

This hook is the place to inject custom flags, wrap the command in your
own script, or modify environment variables before the worker starts, without
forking Dask JobQueue or ESPnet itself.

### Best practices

- Keep provider outputs to be lightweight and serialisable.
- Avoid capturing `self` inside the worker setup function; return a callable that closes over immutable state instead.
- Use the returned results (or per-shard JSONL files in async mode) to implement post-processing such as scoring or manifest generation.

The Provider/Runner architecture keeps experimentation Pythonic while providing a clear path to production-scale clusters.
