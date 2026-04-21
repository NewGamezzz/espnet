---
title: ESPnet3 Provider And Runner
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Provider And Runner

The provider/runner pattern separates environment construction from actual
computation.

## Minimal provider

```python
from espnet3.parallel.env_provider import EnvironmentProvider


class MyProvider(EnvironmentProvider):
    def build_env_local(self):
        return {"dataset": build_dataset(self.config)}

    def build_worker_setup_fn(self):
        config = self.config

        def setup():
            return {"dataset": build_dataset(config)}

        return setup
```

## Minimal runner

```python
from espnet3.parallel.base_runner import BaseRunner


class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, **env):
        return dataset[idx]
```

## Running

```python
provider = MyProvider(cfg)
runner = MyRunner(provider)
results = runner(range(10))
```

The same pattern works in local mode and in Dask-backed mode.

## Notes

- `forward` should stay static
- provider env entries are injected by name
- `build_worker_setup_fn()` is the current worker-side constructor hook
