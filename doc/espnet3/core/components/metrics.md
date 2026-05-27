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

::: note
If you want the full file flow from `metrics.yaml` to `metrics.json`, use the
interactive explorer below.
This page focuses on the metric component contract itself.
:::

<MetricsExplorer />

## Metric contract

Metrics are instantiated from `metrics_config.metrics[*].metric`. Each metric is
called as:

```python
metric(data: Dict[str, Path], test_name: str, output_dir: Path)
```

This is intentionally path-based.
ESPnet3 resolves the requested inputs to SCP paths, but reading and alignment
happen inside the metric.

Typical input:

```python
{
    "ref": Path(".../ref.scp"),
    "hyp": Path(".../hyp.scp"),
}
```

## BaseMetric

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

## iter_inputs() helper

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

Metrics that work directly from file paths can ignore `iter_inputs()` and read
`data[...]` themselves.

## Config aliases

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

This means `measure()` passes:

- `data["ref"]` -> `text.scp`
- `data["hyp"]` -> `hyp.scp`
- `data["prompt"]` -> `prompt.scp`

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

<DocCards :cols="3">
  <DocCard
    title="Metrics configuration"
    desc="See how metric classes and inputs are selected from YAML."
    icon="tabler:settings-2"
    href="../config/metrics.html"
  />
  <DocCard
    title="Metrics stage"
    desc="Return to the stage-level metrics flow."
    icon="tabler:route"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Components overview"
    desc="Return to the full component map."
    icon="tabler:puzzle"
    href="./"
  />
</DocCards>
