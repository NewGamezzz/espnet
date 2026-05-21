# Load checkpoint

Use the model path in the config that owns model construction. For ASR
inference, the common fields are under `model` in `inference.yaml`.

Check these cases:

- local checkpoint path
- packed model directory
- Hugging Face model repo used through publication helpers

Validate shapes before long runs when you changed the model config.

