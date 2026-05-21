# Quick start

This page uses `egs3/mini_an4/asr`. It is the fastest way to see how ESPnet3
 stages fit together.

## Run the recipe

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

## What to expect

- `data/`: prepared manifests and tokenizer assets
- `exp/<exp_tag>/`: checkpoints, logs, and training config snapshots
- `exp/<exp_tag>/inference/<test_name>/`: inference SCP outputs
- `exp/<exp_tag>/inference/metrics.json`: evaluation results

## Next

- Learn the architecture: [System, stage, component](../concepts/system-stage-component.md)
- Inspect stage behavior: [Stages](../stages/index.md)
- Build your own recipe: [First recipe](./first-recipe.md)

