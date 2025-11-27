---
title: ESPnet3 System Entry Point
author:
  name: "Masao Someki"
date: 2025-11-26
---

ESPnet3 training is driven by **System** classes. A System is the entry point
that binds configs to stage functions (prepare, train, decode, score, publish).
This document explains the base interface, how stages are invoked, and how to
extend it for custom tasks.

## 🧠 BaseSystem at a glance

`espnet3.systems.base.system.BaseSystem` implements the common stage hooks:

```python
class BaseSystem:
    def __init__(self, train_config, eval_config=None):
        self.train_config = train_config
        self.eval_config = eval_config
        self.exp_dir = Path(train_config.exp_dir)

    def create_dataset(self): ...
    def collect_stats(self): return collect_stats(self.train_config)
    def train(self): return train(self.train_config)
    def evaluate(self): self.decode(); return self.score()
    def decode(self): return inference(self.eval_config)
    def score(self): return score(self.eval_config)
    def publish(self): ...
```

- The constructor creates `exp_dir` and stores the configs.
- `evaluate()` calls `decode()` then `score()` for convenience.
- Defaults delegate to stage functions in `espnet3.systems.base.{train,inference,score}`.

### ✅ What you typically override

| Method / area   | When you override it                               | Typical user responsibility                     |
| --------------- | --------------------------------------------------- | ----------------------------------------------- |
| `create_dataset`| Custom corpus preparation or manifest generation    | Call recipe-specific `create_dataset.py` logic  |
| `train`         | Extra steps before/after training                   | Tokenizer training, warmup, custom logging      |
| `decode` / `score` | Custom decoding pipeline or metrics             | Hook into custom inference / scoring scripts    |
| `publish`       | Packaging models for release                        | Upload to hubs, export ONNX, create tarballs    |

## How stages are executed

- `run.py` in each recipe instantiates a System and runs selected stages from
  `DEFAULT_STAGES` via `espnet3.utils.stages.run_stages`.
- CLI flags choose stages (`--stage train decode`) and pass per-stage overrides
  parsed by `parse_stage_unknown_args`.
- Each stage receives the System instance, so overrides can be forwarded to
  methods (e.g., `create_dataset` kwargs).

## Extending for a task: example ASRSystem

`espnet3.systems.asr.system.ASRSystem` customises a few hooks:

- **create_dataset(func, kwargs)**: dynamically import and run a dataset
  creation function.
- **train()**: calls `BaseSystem.train()`.
- **train_tokenizer(dataset_dir)**: builds text and runs SentencePiece training.

You can subclass `BaseSystem` to add task-specific steps (e.g., TTS vocoder training).

## When to override

- **create_dataset**: add corpus-specific preparation or manifest generation.
- **train**: insert pre-training steps (tokenizer, LM warmup, checkpoint
  restoration).
- **decode/score**: swap in custom inference or metrics pipeline.
- **publish**: package artifacts, upload to model zoo, or export ONNX.

Keep overrides minimal; rely on the base helpers for standard behavior.

## Config expectations

- `train_config.exp_dir` is required; System will create it.
- `train_config` is passed to training/collect_stats; `eval_config` is passed to
  decode/score. Keep train/eval configs separate to avoid accidental overrides.
- Hydra configs can instantiate models, data modules, callbacks, and parallel
  settings; Systems only orchestrate stages.

## Quick recipe usage

```bash
python run.py \
  --stage train decode score \
  --train_config conf/train.yaml \
  --eval_config conf/eval.yaml
```

This loads configs, builds the System, and runs the chosen stages in order,
using ESPnet3’s Provider/Runner and Lightning plumbing under the hood.
