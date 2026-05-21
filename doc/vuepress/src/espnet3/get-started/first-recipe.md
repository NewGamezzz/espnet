# First recipe

Start from `egs3/TEMPLATE/asr`. The template already matches the current stage
names, config names, and recipe layout.

## Minimum tree

```text
egs3/<recipe>/asr/
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
  run.py
```

## Minimum flow

1. Implement dataset preparation in `dataset/builder.py`.
2. Implement dataset access in `dataset/dataset.py`.
3. Edit `conf/training.yaml`.
4. Run `create_dataset`, `train_tokenizer`, `collect_stats`, and `train`.
5. Run `infer` and `measure`.

See [Recipe directory](../recipe_directory.md) for the current layout.

