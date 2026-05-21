# Troubleshooting

## First checks

- confirm the exact stage name
- confirm the matching config flag
- inspect the resolved config snapshot under `exp/`
- inspect the stage log under the stage output directory

## Frequent failures

- wrong stage name: use `measure`, not `metric`
- wrong config flag: use `--training_config`, not `--train_config`
- missing `inference_dir` for `measure`
- missing tokenizer assets before ASR training
- resolver path mismatch after moving recipe files
- missing optional dependency for metrics, demo, or upload

