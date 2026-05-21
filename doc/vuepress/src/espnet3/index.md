# ESPnet3

ESPnet3 is the Python-first recipe and runtime layer for ESPnet. It uses
named stages, YAML configs, and Lightning-based training to make recipes
easier to read, test, and reuse.

## Start here

- New user: [Get started](./get-started/index.md)
- Need the mental model: [Concepts](./concepts/index.md)
- Want a task-oriented path: [Guides](./guides/index.md)
- Looking up a pipeline stage: [Stages](./stages/index.md)
- Writing recipes or extending internals: [Core reference](./core/index.md)
- Working on the codebase: [Contributing](./contributing/index.md)

## Quick path

Use `egs3/mini_an4/asr` for the smallest end-to-end ASR example.

```bash
cd egs3/mini_an4/asr
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training_asr_transformer.yaml
python run.py --stages infer \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Read [Quick start](./get-started/quick-start.md) before changing configs.

