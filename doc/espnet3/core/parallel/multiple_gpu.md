---
title: ESPnet3 Multi-GPU And Cluster Examples
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Multi-GPU And Cluster Examples

Runner-based workloads can scale through the `parallel` config.

## Local multi-GPU

```yaml
parallel:
  env: local_gpu
  n_workers: 2
  options: {}
```

This uses `dask_cuda.LocalCUDACluster` when available.

## SLURM example

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
```

The selected `options` are forwarded to the corresponding Dask cluster class.

## Related pages

- [Parallel overview](../parallel.md)
- [Provider / runner](./provider_runner.md)
