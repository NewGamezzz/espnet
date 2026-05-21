# Recipe directory

`egs3/TEMPLATE/asr` is the canonical ESPnet3 recipe layout.

## Layout

```text
egs3/<recipe>/<task>/
  run.py
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
    demo.yaml
  dataset/
    __init__.py
    builder.py
    dataset.py
  src/
```

## Responsibilities

- `run.py`: parse CLI flags, load configs, build the system, run stages
- `conf/`: stage-facing configuration files
- `dataset/`: dataset builder and dataset implementation
- `src/`: recipe-local helpers such as output functions or custom modules

## Canonical names

- stages: `create_dataset`, `train_tokenizer`, `collect_stats`, `train`,
  `infer`, `measure`, `pack_model`, `upload_model`, `pack_demo`, `upload_demo`
- flags: `--training_config`, `--inference_config`, `--metrics_config`,
  `--publication_config`, `--demo_config`

## References

- [First recipe](./get-started/first-recipe.md)
- [How configs work](./concepts/how-configs-work.md)
- [Stages](./stages/index.md)

