# Train stage

`train` reads `training.yaml` and writes artifacts under `exp_dir`.

## Run

```bash
python run.py --stages train \
  --training_config conf/training.yaml
```

## Core config fields

| Field | Purpose |
| --- | --- |
| `dataset` | Dataset definitions for train and valid splits. |
| `model` | Model or Lightning wrapper config. |
| `trainer` | Trainer arguments such as devices and precision. |
| `optimizer` / `optimizers` | Optimizer definition(s). |
| `scheduler` / `schedulers` | Scheduler definition(s). |
| `exp_dir` | Checkpoints, logs, and config snapshots. |

## References

- [Training config](../config/training.md)
- [Dataset notes](./train/dataset.md)
- [Dataloader notes](./train/dataloader.md)
- [Optimizer and scheduler notes](./train/optim_scheduler.md)

