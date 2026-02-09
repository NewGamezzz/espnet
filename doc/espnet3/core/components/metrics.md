---
title: ESPnet3 Metrics
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Metrics

This page describes how to implement custom metrics for the `metric` stage.
It focuses on the **metric interface** (`AbsMetric`) and the **I/O contracts**
between inference outputs (`.scp`) and metrics (`metrics.json`).

## What the metric stage passes to a metric

Metrics are instantiated from `metric.yaml` and must implement
`espnet3.components.metrics.abs_metric.AbsMetric`.

At runtime, the metric stage loads `.scp` files under:

- `<inference_dir>/<test_name>/*.scp`

aligns them by utterance ID, and passes them to your metric as a dictionary of
aligned lists:

- `data["utt_id"]`: aligned utterance IDs
- `data["ref"]`, `data["hyp"]`, ...: aligned values loaded from SCPs (strings)

The contract is:

- every list in `data` has the same length as `data["utt_id"]`
- alignment is by utterance ID, not by file order
- if any required SCP is missing or utterance IDs differ across files, metrics fail early

## `AbsMetric` interface

```python
from espnet3.components.metrics.abs_metric import AbsMetric

class MyMetric(AbsMetric):
    def __call__(self, data, test_name, output_dir):
        ...
```

Arguments:

- `data`: `Dict[str, List[str]]` (aligned lists; keys come from `metrics[*].inputs`)
- `test_name`: the current test set name (e.g., `test-clean`)
- `output_dir`: the inference output root (typically `inference_dir`)

Return value:

- must be JSON-serializable (it will be written into `<inference_dir>/metrics.json`)

## SCP format and alignment

Each SCP file is a text file containing one utterance per line:

```text
utt_id VALUE...
```

Only the first whitespace splits the ID from the value, so the value may contain
spaces (useful for full sentences).

Metrics reads the requested SCPs and aligns them by the set of utterance IDs:

- IDs are collected from each SCP and must match across all requested inputs
- IDs are sorted to produce a stable order
- values are gathered in that order to form `data[...]` lists

## Configure a metric in `metric.yaml`

Each metric entry has:

- `metric`: Hydra target for your `AbsMetric` subclass
- `inputs` (optional): which SCP keys to load, and how to name them in `data`

```yaml
inference_dir: exp/my_exp/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test: []

metrics:
  - metric:
      _target_: my_pkg.metrics.WER
    inputs:
      ref: ref   # reads <inference_dir>/<test_name>/ref.scp into data["ref"]
      hyp: hyp   # reads <inference_dir>/<test_name>/hyp.scp into data["hyp"]
```

## Using aliases (mapping `data` keys to SCP filenames)

`inputs` supports aliasing: `alias -> filename`.

This is useful when inference produces multiple outputs such as `hyp0.scp` and
`hyp1.scp`, but your metric expects the key name `hyp`:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.WER
    inputs:
      ref: ref
      hyp: hyp0   # reads hyp0.scp into data["hyp"]
```

## Minimal metric example (WER)

```python
from espnet3.components.metrics.abs_metric import AbsMetric
import jiwer


class WER(AbsMetric):
    def __call__(self, data, test_name, output_dir):
        refs = data["ref"]
        hyps = data["hyp"]
        wer = jiwer.wer(refs, hyps)
        return {"WER": round(wer * 100, 2)}
```

## Audio/file-based metrics (example)

For speech generation tasks, SCP values are often **file paths** (strings).
For example, inference can write:

```text
utt001 <inference_dir>/<test_name>/audio/utt001.wav
utt002 <inference_dir>/<test_name>/audio/utt002.wav
```

Your metric receives those paths as strings in `data["hyp"]` (or any key you map
via `inputs`) and can load audio on demand.

## Related docs

- [Metrics stage](../../stages/metrics.md)
- [Metric config](../../config/metric_config.md)
