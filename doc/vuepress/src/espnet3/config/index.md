# Config overview

Canonical ESPnet3 config files are:

```text
egs3/<recipe>/<task>/conf/
  training.yaml
  inference.yaml
  metrics.yaml
  publication.yaml
  demo.yaml
```

## CLI flags

```bash
python run.py \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml \
  --demo_config conf/demo.yaml
```

## Stage mapping

| Stage | Config flag | Typical file |
| --- | --- | --- |
| `create_dataset` | `--training_config` | `training.yaml` |
| `train_tokenizer` | `--training_config` | `training.yaml` |
| `collect_stats` | `--training_config` | `training.yaml` |
| `train` | `--training_config` | `training.yaml` |
| `infer` | `--inference_config` | `inference.yaml` |
| `measure` | `--metrics_config` | `metrics.yaml` |
| `pack_model` | `--training_config` and `--publication_config` | `training.yaml` and `publication.yaml` |
| `upload_model` | `--publication_config` | `publication.yaml` |
| `pack_demo` | `--demo_config` | `demo.yaml` |
| `upload_demo` | `--demo_config` | `demo.yaml` |

## Pages

- [Training config](./training.md)
- [Inference config](./inference.md)
- [Metrics config](./metrics.md)
- [Publication config](./publication.md)
- [Demo config](./demo.md)
- [Resolvers](./resolvers.md)

