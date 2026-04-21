---
title: ESPnet3 System-Specific Stages
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 System-Specific Stages

This page is for people who develop ESPnet3 **System** classes, not for
day-to-day recipe users.

ESPnet3 stage execution is driven by a **System** class, typically a subclass
of `espnet3.systems.base.system.BaseSystem`. The System provides stage methods
such as `train()`, `infer()`, and `measure()`, and it can also define
**system-specific stages** that are not part of the generic pipeline.

This page covers two extension patterns:

1. customizing the behavior of an existing stage such as `train`
2. adding a brand-new stage and exposing it from the recipe `run.py`

## At a glance

| Goal | What to change | Typical location | User-facing stage name |
| --- | --- | --- | --- |
| Change how `train` or `collect_stats` works for a task | Override system internals | `espnet3/systems/<task>/...` | unchanged |
| Add a recipe-only extra step | Add a new system method and expose it in `run.py` | `egs3/<recipe>/<task>/src/system.py` | new stage name |
| Reuse helper code from a custom stage | Split method body into helpers | `egs3/<recipe>/<task>/src/stages.py` | unchanged |
| Add a stage-specific config file | Extend parser, load config, store on system, add `required_configs` rule | recipe `run.py` + `conf/` | unchanged or new |

The shortest rule of thumb is:

- follow the `GanTTS` pattern when you want to customize an existing stage
- use `src/system.py` when you want a recipe-local new stage
- keep stage parameters in YAML, not in ad-hoc CLI flags

## What is a system-specific stage?

A system-specific stage is any method on your System that you add to the stage
list in your recipe `run.py`.

Examples:

- tokenizer training (`train_tokenizer`)
- task-specific preprocessing (`prepare_labels`, `dump_features`)
- export (`export_onnx`, `export_torchscript`)
- custom evaluation (`evaluate_custom`)
- recipe-specific preparation (`prepare_alignment`, `cache_prompts`)

## How stages are discovered and executed

Recipe `run.py` defines the available stage names and passes them to the
argument parser through `--stages`. Internally, stage names are resolved and
then invoked by `espnet3.utils.stages_utils.run_stages()` as methods on the
system instance.

In other words:

- if `--stages foo` is requested, the System must implement `def foo(self): ...`
- stage settings should live in YAML configs
- stages are called without extra CLI arguments

That is why stage logic belongs in the system class rather than in `run.py`.

## When you need system-specific behavior

You usually need this when the generic ASR-style workflow is not enough.

Common reasons are:

- training logic depends on the model family
- one task needs to override the internal implementation of `train`
- one recipe needs an extra preprocessing or export step
- a stage needs to call recipe-local helper code under `src/`

If the change applies to many recipes of the same task, it usually belongs
under `espnet3/systems/<task>/`.

If the change is recipe-local, it is usually clearer to put it under:

```text
egs3/<recipe>/<task>/src/
```

and import that system class from the recipe `run.py`.

## `GanTTS` as a reference

`GanTTS` is the main example of system-specific behavior in the current code
base.

The reusable implementation lives in:

```text
espnet3/systems/tts/system.py
espnet3/systems/tts/gan_trainer.py
```

The important point is that `GanTTS` does not introduce a new user-facing stage
name. Instead:

- stage names stay standard: `collect_stats`, `train`, `infer`, ...
- `TTSSystem` overrides how `collect_stats` and `train` are executed
- when the instantiated model is a GAN model, the system automatically uses
  `GANTTSLightningTrainer`

In [espnet3/systems/tts/system.py](/mnt/c/Users/might/Documents/git/espnets/espnet3/espnet3/systems/tts/system.py:37),
`TTSSystem._build_trainer()` checks the model type and switches trainer
construction:

```python
if isinstance(model, AbsGANESPnetModel):
    return build_gan_trainer(config, model)
```

Then [espnet3/systems/tts/gan_trainer.py](/mnt/c/Users/might/Documents/git/espnets/espnet3/espnet3/systems/tts/gan_trainer.py:13)
wraps the training logic in `GANTTSLightningTrainer`.

This is the pattern to follow when users should still run:

```bash
python run.py --stages train --training_config conf/training.yaml
```

but the internals of that stage must change for a specific task or model.

## Adding a new stage in your recipe

If you need a genuinely new operation, the normal pattern is:

1. implement a method on your system
2. expose that stage name in the recipe `run.py`
3. put stage settings in YAML
4. run it through `python run.py --stages ...`

### Where to put the code

For recipe-local work, a practical layout is:

```text
egs3/my_recipe/asr/
  run.py
  conf/
    training.yaml
  src/
    system.py
    stages.py
```

The simplest option is to define a recipe-local system subclass in
`src/system.py` and place the custom stage method there.

Example:

```python
from espnet3.systems.asr.system import ASRSystem


class MyASRSystem(ASRSystem):
    def prepare_alignment(self):
        cfg = self.training_config.prepare_alignment
        output_dir = cfg.output_dir
        # call any recipe-local helper here
```

If the body gets large, you can move the implementation to `src/stages.py`:

```python
from espnet3.systems.asr.system import ASRSystem
from src.stages import prepare_alignment_stage


class MyASRSystem(ASRSystem):
    def prepare_alignment(self):
        prepare_alignment_stage(self.training_config)
```

### Expose it in your recipe `run.py`

Add the stage name to the stage list and instantiate your custom system:

```python
from egs3.TEMPLATE.asr.run import build_parser, main
from espnet3.utils.stages_utils import parse_cli_and_stage_args
from src.system import MyASRSystem

DEFAULT_STAGES = [
    "create_dataset",
    "prepare_alignment",
    "collect_stats",
    "train",
    "infer",
    "measure",
]

parser = build_parser(stages=DEFAULT_STAGES)
args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

main(
    args=args,
    system_cls=MyASRSystem,
    stages=DEFAULT_STAGES,
)
```

### Run it

```bash
python run.py \
  --stages prepare_alignment \
  --training_config conf/training.yaml
```

## Config requirements and adding new configs

Which stages require which config files is not inferred from the System class.
It is enforced by each recipe `run.py`.

The current runner pattern is:

- CLI flags load optional configs such as `--training_config`,
  `--inference_config`, `--metrics_config`, and `--publication_config`
- `run.py` defines a mapping from stage name to required config object
- if a requested stage is missing its required config, `run.py` raises early

Typical starting point: read the template runner at
[egs3/TEMPLATE/asr/run.py](/mnt/c/Users/might/Documents/git/espnets/espnet3/egs3/TEMPLATE/asr/run.py:27).

### Checklist when a new stage needs its own config

When a new system-specific stage needs its own config instead of reusing
`training.yaml`, `inference.yaml`, `metrics.yaml`, or `publication.yaml`,
follow this checklist.

1. Add a new CLI flag and load the config.

```python
parser.add_argument(
    "--export_config",
    default=None,
    type=Path,
    help="Hydra config for export-related stages.",
)

export_config = (
    None
    if args.export_config is None
    else load_config_with_defaults(args.export_config)
)
```

2. Pass it into your System.

```python
system = system_cls(
    training_config=training_config,
    inference_config=inference_config,
    metrics_config=metrics_config,
    publication_config=publication_config,
    export_config=export_config,
)
```

3. Store it on the System.

```python
class MySystem(BaseSystem):
    def __init__(self, *, export_config=None, **kwargs):
        super().__init__(**kwargs)
        self.export_config = export_config
```

4. Enforce "stage requires config" in `run.py`.

```python
required_configs = {
    "create_dataset": training_config,
    "collect_stats": training_config,
    "train": training_config,
    "infer": inference_config,
    "measure": metrics_config,
    "pack_model": (training_config, publication_config),
    "upload_model": publication_config,
    "export": export_config,
}

missing = [
    s
    for s in stages_to_run
    if s in required_configs
    and (
        any(cfg is None for cfg in required_configs[s])
        if isinstance(required_configs[s], tuple)
        else required_configs[s] is None
    )
]
```

5. Add docs and an example `conf/*.yaml`.

Create a config file under `egs3/<recipe>/<task>/conf/` and link it from the
relevant stage doc under `doc/vuepress/src/espnet3/stages/`.

## How to configure and run a custom stage

Custom stages should usually read settings from `training.yaml`.

Example config:

```yaml
prepare_alignment:
  output_dir: ${exp_dir}/alignment
  use_teacher_forcing: true
```

Example system method:

```python
class MyASRSystem(ASRSystem):
    def prepare_alignment(self):
        cfg = self.training_config.prepare_alignment
        output_dir = cfg.output_dir
        use_teacher_forcing = cfg.use_teacher_forcing
        # run recipe-specific logic here
```

Then run it like any other stage:

```bash
python run.py \
  --stages prepare_alignment train \
  --training_config conf/training.yaml
```

Keep the stage method signature simple:

```python
def my_stage(self):
    ...
```

Do not add stage-specific CLI arguments unless the stage truly needs a separate
config object.

## Recommended conventions

- use short, verb-style snake_case stage names such as `train_tokenizer` or
  `export_onnx`
- keep stages idempotent when possible
- reuse standard stages such as `create_dataset`, `collect_stats`, `train`,
  `infer`, `measure`, `pack_model`, and `upload_model` unless a new stage is
  genuinely needed
- follow the `GanTTS` pattern when you only need to customize the internals of
  an existing stage

## Related docs

- [Systems overview](../core/systems.md)
- [Stage configs](../config/index.md)
- [Train](./train.md)
- [Inference](./inference.md)
- [Measure](./measure.md)
- [Trainer](../core/components/trainer.md)
