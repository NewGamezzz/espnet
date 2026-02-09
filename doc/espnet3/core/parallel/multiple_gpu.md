---
title: ESPnet3 - Multi-GPU and Multi-Node Execution
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🚀 ESPnet3: Multi-GPU and Multi-Node Execution

This page is **example-driven**. It focuses on multi-GPU / multi-node execution
for inference-like workloads implemented with `EnvironmentProvider` +
`BaseRunner`.

See also:

- [Provider / Runner](./provider_runner.md)
- [ESPnet3 Parallel](../parallel.md)

### 1. Inference or evaluation with runners

For multi-GPU inference, decoding, or scoring jobs ESPnet3 provides the
`BaseRunner` class.  A runner processes indices in parallel while an
`EnvironmentProvider` constructs datasets and models on each worker.

```python
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.parallel import set_parallel
from espnet3.parallel.inference_provider import InferenceProvider

class DecodeProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg):
        return load_eval_split(cfg.dataset)

    @staticmethod
    def build_model(cfg):
        model = build_model(cfg.model)
        return model.to(cfg.model.device)

class DecodeRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset, model, beam_size=5):
        sample = dataset[idx]
        return {
            "utt_id": sample["utt_id"],
            "hyp": model.decode(sample, beam_size=beam_size),
        }

provider = DecodeProvider(cfg, params={"beam_size": 8})
runner = DecodeRunner(provider)

set_parallel(cfg.parallel)  # same runner works locally or on SLURM
results = runner(range(len(eval_set)))
```

Workers automatically receive their own dataset/model instances.

#### Local: use multiple GPUs on one machine


Use `env: local_gpu` (requires `dask_cuda`) and set `n_workers` to the number
of GPUs you want to use. ESPnet3 consumes only these keys:

- `parallel.env`
- `parallel.n_workers`
- `parallel.options`

```yaml
parallel:
  env: local_gpu
  n_workers: 2
  options: {}  # passed through to dask_cuda.LocalCUDACluster(...)
```

#### SLURM: submit a multi-GPU job <span class='small-bracket'>(one GPU per worker)</span>


Use `env: slurm` and set `n_workers` to the total number of GPUs you want to
use. The `options` keys are passed to `dask_jobqueue.SLURMCluster(...)`, so you
should include whatever your cluster requires; a common minimal set is below.

More generally, `parallel.options` is forwarded as keyword arguments to the
selected Dask cluster class (e.g., `dask_jobqueue.SLURMCluster`, `PBSCluster`,
...). For the full list of supported cluster types and their constructor
arguments, see:

- [Dask Jobqueue cluster API](https://jobqueue.dask.org/en/latest/clusters-api.html)

If you want SLURM-specific flags beyond this example (or want to use a
non-SLURM backend), follow the per-cluster links from that page and use the
documented constructor arguments under `parallel.options`.

```yaml
parallel:
  env: slurm
  n_workers: 4
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 32GB
    walltime: "04:00:00"
    job_extra_directives:
      - "--gres=gpu:1"
      # Add any site-specific directives here (partition, account, constraint, ...)
    job_script_prologue:
      - "source ./path.sh"
    local_directory: "./_dask_tmp"
```

In this pattern, **one Dask worker == one GPU**. Each worker runs the same
runner code and processes a shard of indices.

### 2. Local vs. cluster performance

The refactored runner supports three modes: local, synchronous cluster jobs, and
asynchronous cluster submissions.  The same decoding runner was benchmarked in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353) on an
A40 GPU with the OWSM-V4 medium (1B) model over 1,000 LibriSpeech test-clean
utterances:

| Environment  | #GPUs | Wall time (s) |
| ------------ | ----- | ------------- |
| local        | 1     | 1220          |
| local        | 2     | 695           |
| slurm / sync | 1     | 1336          |
| slurm / sync | 2     | 669           |
| slurm / sync | 4     | 369           |

In synchronous mode the driver waits for all workers to finish, whereas in
asynchronous mode the submission script becomes a lightweight dispatcher and the
worker jobs continue even if the driver exits early.
