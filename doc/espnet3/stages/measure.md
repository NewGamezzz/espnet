---
title: 📘 ESPnet3 Measure Stage
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Measure Stage

This page describes the current measurement flow in ESPnet3.

The stage entrypoint is:

- `espnet3.systems.base.metric.measure`

Custom metrics are implemented with:

- `espnet3.components.metrics.base_metric.BaseMetric`

`measure` reads inference outputs under `inference_dir/<test_name>/`, computes
one or more metrics, and writes a single summary file:

```text
<inference_dir>/metrics.json
```

## Quick usage

```bash
python run.py --stages measure --metrics_config conf/metrics.yaml
```

The template config is:

- `egs3/TEMPLATE/asr/conf/metrics.yaml`


## What gets passed to a metric

Each entry in `metrics.yaml` is handled like this:

1. instantiate `metrics_config.metrics[*].metric`
2. resolve input SCP paths for one `test_name`
3. call the metric class



`measure()` does not preload SCP contents into lists. It resolves file paths and
passes them directly to each metric.

Example config:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
```

For `test_name = test-clean`, `measure()` instantiates that class and passes:

```python
{
    "ref": Path("exp/.../inference/test-clean/ref.scp"),
    "hyp": Path("exp/.../inference/test-clean/hyp.scp"),
}
```

So the metric contract is:

- `data`: `Dict[str, Path]`
- `test_name`: current test-set name
- `output_dir`: the root `inference_dir`

That means the metric class itself reads SCP contents.

For aligned SCP inputs, the normal implementation pattern is
`BaseMetric.iter_inputs(...)`.

## Sample config

```yaml
recipe_dir: .
exp_tag:
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
      clean_types:

  - metric:
      _target_: espnet3.systems.asr.metrics.cer.CER
      ref_key: ref
      hyp_key: hyp
      clean_types:
```

## Test-set resolution

`measure()` resolves test sets in this order:

1. If `metrics_config.dataset.test` exists, use each item's `name`.
2. Otherwise, scan `metrics_config.inference_dir` for subdirectories.

This is why the current template no longer requires duplicating test-set
definitions in `metrics.yaml`.

## Inputs and SCP filenames

Each metric can receive inputs in two ways.

If `inputs` is defined in config:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.MyMetric
    inputs:
      ref: ref
      hyp: hyp
      prompt: prompt
```

then ESPnet3 resolves:

- `data["ref"] -> <test_name>/ref.scp`
- `data["hyp"] -> <test_name>/hyp.scp`
- `data["prompt"] -> <test_name>/prompt.scp`

If `inputs` is omitted, `measure()` falls back to the metric instance's
`ref_key` and `hyp_key`.

## Outputs

The summary file format is:

```text
<inference_dir>/
  metrics.json
  test-clean/
    ref.scp
    hyp.scp
    wer_alignment
    cer_alignment
```

`metrics.json` is keyed by metric class path, then by test-set name.

## Related pages

- [Metrics configuration](../config/measure_config.md)
- [Custom metrics](../core/components/metrics.md)
