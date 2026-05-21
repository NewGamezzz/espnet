# How configs work

ESPnet3 uses one config file per major pipeline area.

## Config files

- `training.yaml`: dataset prep, tokenizer training, stats, and training
- `inference.yaml`: model loading and inference outputs
- `metrics.yaml`: metric definitions and `inference_dir`
- `publication.yaml`: pack and upload settings
- `demo.yaml`: demo packaging and Space upload settings

## Load flow

`egs3/TEMPLATE/asr/run.py` loads configs with `load_and_merge_config()`, applies
experiment context, then resolves the merged configs before stage execution.

## Important patterns

- `${exp_dir}` and `${recipe_dir}` interpolate shared paths
- `_target_` selects a Python class or function to instantiate
- stage settings live in YAML, not ad-hoc CLI flags

## Current CLI flags

```bash
python run.py \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml \
  --demo_config conf/demo.yaml
```

