---
title: ESPnet3 Data Preparation With Providers And Runners
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Data Preparation With Providers And Runners

The same provider/runner infrastructure can be used for dataset preparation or
heavy preprocessing jobs.

Typical pattern:

1. provider builds datasets and helper objects
2. runner processes indices
3. outputs are written as artifacts or metadata

This is useful when preparation work should scale from local runs to Dask
clusters without rewriting the Python logic.

## Skeleton

```python
class PrepProvider(EnvironmentProvider):
    def build_env_local(self):
        return {"dataset": load_dataset(self.config)}

    def build_worker_setup_fn(self):
        config = self.config
        return lambda: {"dataset": load_dataset(config)}


class PrepRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, **env):
        sample = dataset[idx]
        prepare_sample(sample)
        return {"utt_id": sample["utt_id"]}
```
