# Metrics config

`metrics.yaml` controls the `measure` stage.

Common top-level areas are:

- `dataset.test` or test directories under `inference_dir`
- `metrics`
- `inference_dir`

Metric classes must inherit from `BaseMetric`.

