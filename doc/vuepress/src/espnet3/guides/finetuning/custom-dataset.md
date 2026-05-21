# Custom dataset for finetuning

Create the dataset module under `egs3/<recipe>/<task>/dataset/`. Reuse
`espnet3/components/data` helpers instead of writing ad-hoc loaders.

Then update:

- `training.yaml`
- `inference.yaml` when test data differs
- `metrics.yaml` when `inference_dir` or test sets differ

