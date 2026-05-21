# Measure stage

`measure` is the canonical evaluation stage name in ESPnet3.

## Run

```bash
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

## What it reads

- `metrics.yaml`
- `inference_dir/<test_name>/*.scp`

## What it writes

- `inference_dir/metrics.json`

Use `BaseMetric` subclasses in `metrics.yaml`.

