# Config for finetuning

Start from the training config that matches the checkpoint family, then change
only the fields that define the new dataset and optimization behavior.

Typical edits:

- dataset entries
- learning rate
- warmup or scheduler settings
- epoch count
- batch size or accumulation

