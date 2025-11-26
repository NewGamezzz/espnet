---
title: ESPnet3 Recipe Directory Layout
---

ESPnet3 recipes live under `egs3/` and follow a consistent structure so you can
reuse tooling across corpora and tasks. This page explains the directory
layout, the role of `run.py`, and where to place configs and custom code.

## Directory structure

```
egs3/
  TEMPLATE/         # Minimal scaffold to copy for new recipes
    asr/
      run.py        # Stage runner wiring
      readme.md     # Quickstart for the template
  librispeech_100/  # Example corpus
    asr/
      run.py        # Entry point (imports TEMPLATE logic)
      run.sh        # Optional helper scripts (legacy style)
      conf/         # Hydra/YAML configs and tuning variants
      data/         # Local data/manifests produced by stages
      exp/          # Experiment outputs (checkpoints, stats)
      logs/         # Stage logs
      src/          # Recipe-specific Python helpers
      readme.md     # Recipe-specific instructions
```

- **TEMPLATE** holds a working skeleton (Python stages, parser wiring). Copy it
  when starting a new recipe.
- **corpus/task** folders (e.g., `librispeech_100/asr`) customise configs and
  optional helpers but reuse the same stage runner.

## run.py: stage driver

`run.py` is the single entry point. It loads configs, instantiates a System
class, and executes the requested stages:

```python
from egs3.TEMPLATE.asr1.run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stage_configs, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=ASRSystem, stages=stages_to_run, stage_configs=stage_configs)
```

- **Stages** (`DEFAULT_STAGES`) define the lifecycle: `create_dataset`,
  `train_tokenizer`, `collect_stats`, `train`, `evaluate`, `decode`, `score`,
  `publish`.
- CLI flags select stages (`--stage train decode`) and configs
  (`--train_config conf/train.yaml`, `--eval_config conf/eval.yaml`).
- `--*_overrides` arguments (Hydra syntax) can patch values per stage.

## Configs and outputs

- **conf/** stores YAML configs (train/eval, tuning variants).
- **data/** holds prepared datasets/manifests produced by stages.
- **exp/** is the experiment root for checkpoints, averaged models, and stats.
- **logs/** captures stdout/stderr per stage.
- **src/** is optional for recipe-specific Python helpers; import them from
  `run.py` or configs as needed.

## Creating a new recipe

1) Copy `egs3/TEMPLATE/<task>` into `egs3/<your_corpus>/<task>`.  
2) Add configs under `conf/` (model, trainer, parallel, dataloader).  
3) Point `run.py` to your `System` subclass (or keep `ASRSystem` if applicable).  
4) Run stages with `python run.py --stage create_dataset train --train_config conf/train.yaml`.  

With this layout, every recipe shares the same stage driver while keeping data,
configs, and outputs contained per corpus/task.
