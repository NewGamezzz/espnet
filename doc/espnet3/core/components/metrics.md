---
title: ESPnet3 Metrics
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Metrics

This page describes how to implement custom metrics for the current ESPnet3
measurement flow.

The base class is:

- `espnet3.components.metrics.base_metric.BaseMetric`

The stage entrypoint is:

- `espnet3.systems.base.metric.measure`

## Current metric contract

Metrics are instantiated from `metrics_config.metrics[*].metric`. Each metric is
called as:

```python
metric(data: Dict[str, Path], test_name: str, output_dir: Path)
```

This is intentionally path-based. ESPnet3 resolves required inputs to SCP paths,
but it does not preload them into aligned lists.

### What `data` contains

Typical ASR example:

```python
{
    "ref": Path("exp/asr_run/inference/test-clean/ref.scp"),
    "hyp": Path("exp/asr_run/inference/test-clean/hyp.scp"),
}
```

Another metric can request different files:

```python
{
    "ref": Path(".../ref.scp"),
    "hyp": Path(".../hyp.scp"),
    "prompt": Path(".../prompt.scp"),
}
```

So the important distinction is:

- `data` contains file paths
- reading and alignment happen inside the metric, usually through helpers

## `BaseMetric`

```python
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class MyMetric(BaseMetric):
    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        ...
```

Return values must be JSON-serializable because `measure()` stores them in
`<inference_dir>/metrics.json`.

## `iter_inputs()` is the standard helper

For aligned SCP text inputs, use:

```python
for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
    ref = row["ref"]
    hyp = row["hyp"]
```

`iter_inputs()`:

- opens the requested SCP files
- reads them in file order
- checks that utterance IDs match
- yields one aligned row at a time

This is the normal pattern for text metrics such as WER and CER.

## Path-based metrics vs content-consuming metrics

There are two common patterns.

### 1. Metrics that stream SCP contents

WER/CER style metrics use `iter_inputs()`:

```python
refs = []
hyps = []
for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
    refs.append(row[self.ref_key])
    hyps.append(row[self.hyp_key])
```

### 2. Metrics that consume paths directly

Some metrics may call external tools or inspect directories directly:

```python
ref_scp = data["ref"]
hyp_scp = data["hyp"]
subprocess.run(["some_tool", str(ref_scp), str(hyp_scp)], check=True)
```

In those cases, no SCP parsing helper is required.

## Configuring aliases

`measure()` accepts either:

- implicit keys via the metric's `ref_key` / `hyp_key`
- explicit aliases through `inputs`

Example:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.CustomMetric
    inputs:
      ref: text
      hyp: hyp
      prompt: prompt
```

This means:

- `data["ref"]` points to `text.scp`
- `data["hyp"]` points to `hyp.scp`
- `data["prompt"]` points to `prompt.scp`

## Example: text metric

```python
from pathlib import Path
from typing import Dict

import jiwer

from espnet3.components.metrics.base_metric import BaseMetric


class SimpleWER(BaseMetric):
    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        refs = []
        hyps = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(row[self.ref_key])
            hyps.append(row[self.hyp_key])
        return {"WER": jiwer.wer(refs, hyps) * 100}
```

## Example: path-driven metric

```python
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class FileCountMetric(BaseMetric):
    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        with open(data["hyp"], encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return {"num_hypotheses": float(count)}
```

## Related pages

- [Measure stage](../../stages/measure.md)
- [Measure configuration](../../config/measure_config.md)
