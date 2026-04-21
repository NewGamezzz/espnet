---
title: ESPnet3 Measure Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Measure Configuration

This page describes the current config used by:

```bash
python run.py --stages measure --metrics_config conf/metrics.yaml
```

## Minimum required keys

Required:

- `inference_dir`
- `metrics`

Optional:

- `dataset.test`
- `recipe_dir`
- `exp_tag`
- `exp_dir`

`dataset.test` is optional because `measure()` can discover test-set names from
`inference_dir`.

## Minimal example

```yaml
inference_dir: ${recipe_dir}/exp/inference_beam5/inference

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
      clean_types:
```

## Config sections

| Section | Purpose |
| --- | --- |
| `recipe_dir`, `exp_tag`, `exp_dir`, `inference_dir` | Output and experiment naming |
| `dataset.test` | Optional explicit test-set names |
| `metrics` | Metric definitions to instantiate and run |

## Default values

| Key | Default value |
| --- | --- |
| `recipe_dir` | `.` |
| `exp_tag` | empty |
| `exp_dir` | `${recipe_dir}/exp/${exp_tag}` |
| `inference_dir` | `${exp_dir}/${self_name:}` |
| `dataset.test` | omitted |

## Naming behavior

If `inference_config` is also loaded in the same `run.py` invocation, measure
inherits the inference-side context and uses the same `inference_dir`.

Example 1: measure inherits `inference_dir`.

`inference.yaml`:

```yaml
exp_tag: inference_beam5
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

`metrics.yaml`:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
```

Run:

```bash
python run.py \
  --stages infer measure \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

In this case, measure uses:

```text
exp/inference_beam5/inference/
```

Example 2: measure uses its own `metrics.yaml`.

```yaml
exp_tag: measure_beam5
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${recipe_dir}/exp/inference_beam5/inference

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
```

Run:

```bash
python run.py \
  --stages measure \
  --metrics_config conf/metrics.yaml
```

In this case, measure does not inherit inference context, so `inference_dir`
must be set in `metrics.yaml`.

## Optional explicit test sets

If you want to pin the measured test sets explicitly, add:

```yaml
dataset:
  test:
    - name: test-clean
    - name: test-other
```

If you omit this block, the stage scans:

```text
<inference_dir>/
  test-clean/
  test-other/
```

and computes metrics for all test-set directories under `inference_dir`.

## Metric entries

Each entry in `metrics` has:

- `metric`: a Hydra target for a `BaseMetric` subclass
- `inputs`: optional alias-to-filename mapping

Example:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.MyMetric
    inputs:
      ref: ref
      hyp: hyp
      prompt: prompt
```

`measure()` resolves these paths and passes them to the metric class
`__call__` method:

```python
{
    "ref": Path("<inference_dir>/<test_name>/ref.scp"),
    "hyp": Path("<inference_dir>/<test_name>/hyp.scp"),
    "prompt": Path("<inference_dir>/<test_name>/prompt.scp"),
}
```

In other words, the metric class receives SCP paths, not loaded string lists.
The metric implementation should read those SCP files and compute the metric.

See [Custom metrics](../core/components/metrics.md) for metric class details.

Typical implementation:

```python
class MyMetric(BaseMetric):
    def __call__(self, data, test_name, output_dir):
        for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
            ref = row["ref"]
            hyp = row["hyp"]
            ...
        return {"score": 0.0}
```

If `inputs` is omitted, the stage falls back to the metric instance's
`ref_key` and `hyp_key`.

## Output file

Results are written to:

```text
<inference_dir>/metrics.json
```

## Related pages

- [Measure stage](../stages/measure.md)
- [Custom metrics](../core/components/metrics.md)
