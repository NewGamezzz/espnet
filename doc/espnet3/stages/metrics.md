---
title: 📘 ESPnet3 Metric Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Metric Stage

This document explains the **metrics stage** in ESPnet3, implemented in:

* `espnet3.systems.base.metric.metric`
* `espnet3.components.metrics.abs_metric.AbsMetric`

Metrics read the `ref.scp` and `hyp.scp` files produced by inference and
writes a `metrics.json` summary.

For the full metric interface (how `AbsMetric` is called, how SCPs are aligned,
and how to implement custom metrics), see:

- [ESPnet3 Metrics](../core/components/metrics.md)

## Quick usage

### Run

```bash
python run.py --stages metric --metric_config conf/metric.yaml
```

### Configure (in metric.yaml)

Keep the core settings in `metric.yaml`. For the full list, see
[Metric configuration](../core/config/metrics.md).

| Config section | Description |
| -------------- | ----------- |
| `dataset` | Dataset organizer and test splits. Metrics use this to iterate test set names. |
| `metrics` | List of metric definitions. Each entry specifies `metric` and optional `inputs`. |
| `inference_dir` | Location of `.scp` files under `inference_dir/<test_name>/`. |

<!-- TODO(masao): update this section after the PR that removes the dataset dependency in metric(). -->

### Outputs

Metrics write:

```text
<inference_dir>/metrics.json
```

## Developer Notes

### 🧩 Config fields used during metrics

A minimal `metric_config` for metrics looks like:

```yaml
inference_dir: exp/asr_example/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test-clean
      dataset:
        _target_: ...
    - name: test-other
      dataset:
        _target_: ...

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
    inputs:
      ref: ref
      hyp: hyp
```

### Metric interface

Metric classes are defined as `AbsMetric` subclasses and are instantiated from
`metric.yaml`. See the core documentation for details:

- [ESPnet3 Metrics](../core/components/metrics.md)
