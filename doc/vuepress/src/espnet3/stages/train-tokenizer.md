# Train tokenizer

`train_tokenizer` is an ASR-specific stage implemented by `ASRSystem`.

## Run

```bash
python run.py --stages train_tokenizer \
  --training_config conf/training.yaml
```

## What it uses

- `training_config.tokenizer`
- `training_config.tokenizer.text_builder.func`
- `training_config.tokenizer.save_path`

The stage is skipped when tokenizer artifacts already exist.

