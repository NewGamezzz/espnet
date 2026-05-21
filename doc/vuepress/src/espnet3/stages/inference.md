# Inference stage

The canonical inference entrypoint is `espnet3.systems.base.inference.infer`.

## Run

```bash
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```

## What it uses

- `inference.yaml`
- `provider`
- `runner`
- `input_key`
- `output_fn`
- `inference_dir`

## What it writes

For each test set:

```text
<inference_dir>/<test_name>/*.scp
```

Use [Inference config](../config/inference.md) for the config-level reference.

