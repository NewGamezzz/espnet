---
title: ESPnet2 to ESPnet3 Migration
author:
  name: "Masao Someki"
date: 2026-02-02
---

# ESPnet2 to ESPnet3 Migration

This document is a practical guide for migrating an **ESPnet2 recipe** under
`egs2/` to an **ESPnet3 recipe** under `egs3/`.

It is written for recipe authors who want:

- a repeatable migration flow,
- concrete examples you can copy/paste, and
- pointers to the “real” docs for config details (so this page stays readable).

If you are new to ESPnet3, read these first:

- [Getting started](../get_started.md)
- [Recipe directory layout](../recipe_directory.md)
- [Systems](./systems.md)
- [Naming convention](./naming_convention.md)

## TL;DR: the mental model shift

ESPnet2 recipes are usually driven by shell (`run.sh` → `asr.sh` → stage numbers).
ESPnet3 recipes are driven by Python (`run.py` → `System` → stage methods), and
nearly all knobs move into YAML.

## Reference recipes in this repository

Use these as your “diff viewer” while migrating.

### ESPnet2: LibriSpeech 100h

`egs2/librispeech_100/asr1` (abridged)

```text
egs2/librispeech_100/asr1/
├── run.sh
├── asr.sh
├── local/
│   └── data.sh
└── conf/
    ├── train_asr.yaml
    └── decode_asr.yaml
```

Key points:

- `local/data.sh` downloads + prepares Kaldi-style data directories
- `conf/train_asr.yaml` and `conf/decode_asr.yaml` are consumed by `asr.sh`
- stage behavior is largely in shell (`asr.sh`) rather than Python

### ESPnet3: LibriSpeech 100h

`egs3/librispeech_100/asr` (abridged)

```text
egs3/librispeech_100/asr/
├── run.py
├── run.sh
├── conf/
│   ├── infer.yaml
│   ├── metric.yaml
│   ├── publish.yaml
│   └── tuning/
│       └── train_*.yaml
└── src/
    ├── create_dataset.py
    ├── dataset.py
    └── tokenizer.py
```

Key points:

- `src/create_dataset.py` replaces most of `local/data.sh`
- `src/dataset.py` replaces “Kaldi data dir parsing” with a Torch dataset
- `conf/*.yaml` drive the stage behavior

### ESPnet3: TEMPLATE (what you copy when starting a recipe)

`egs3/TEMPLATE/asr` (abridged)

```text
egs3/TEMPLATE/asr/
├── run.py
├── readme.md
├── conf/
│   ├── train.yaml
│   ├── infer.yaml
│   ├── metric.yaml
│   ├── publish.yaml
│   └── demo.yaml
└── src/
    └── ...
```

This is the intended “starter kit” for new recipes:

- copy it into `egs3/<corpus>/<system>/`
- then replace config + `src/` code gradually

### ESPnet3: mini_an4 (useful for integration tests)

`egs3/mini_an4/asr` is a small recipe that is convenient for end-to-end checks.
It also shows a manifest-based dataset pattern (TSV manifests under `data/`).

See also the stage docs:

- [Create dataset](../stages/create-dataset.md)
- [Train](../stages/train.md)
- [Inference](../stages/inference.md)
- [Metric](../stages/metrics.md)
- [Demo](../stages/demo.md)

## Stage name / config name note (important)

The **source of truth** for stage names is the System methods:
`espnet3/systems/base/system.py`.

In ESPnet3 docs and naming conventions:

- stage name is `metric`
- config is `metric_config`
- config file is typically `conf/metric.yaml`

If you see older recipes using `measure`/`measure.yaml`, treat that as an older
name and prefer `metric` going forward.

## Migration process (recommended flow)

This is the flow we recommend for migrating any `egs2/<corpus>/<task>` recipe.
The LibriSpeech 100h examples are called out explicitly.

### 1) Pick the recipe and choose the System

Start by deciding the ESPnet3 System class:

- ASR: `espnet3.systems.asr.system.ASRSystem`
- (others: see [Systems](./systems.md))

You should rarely need a new System on day 1. Start by reusing the closest
existing system and only extend it when you truly need a new stage or new wiring.

If you need additional stages, read:

- [System-specific stages](../stages/system-specific.md)

### 2) Create the new recipe directory (copy TEMPLATE)

Copy the template directory:

```bash
cp -r egs3/TEMPLATE/asr egs3/<your_corpus>/<your_system>
```

Then edit:

- `egs3/<your_corpus>/<your_system>/run.py` (wire the System class)
- `egs3/<your_corpus>/<your_system>/conf/*.yaml` (paths + settings)
- `egs3/<your_corpus>/<your_system>/src/*` (dataset + helpers)

For LibriSpeech 100h, compare with:

- `egs2/librispeech_100/asr1`
- `egs3/librispeech_100/asr`

### 3) Port dataset preparation: `local/data.sh` -> `src/create_dataset.py`

ESPnet2 often does:

- download archives
- extract
- build Kaldi-style data dirs (`data/<split>/{wav.scp,text,utt2spk,...}`)

ESPnet3 expects you to implement one of these patterns:

#### Pattern A: “native layout” dataset (recommended when possible)

Keep the corpus’s native directory layout and implement a dataset loader that
reads it directly.

LibriSpeech 100h does exactly this:

- ESPnet2: `egs2/librispeech_100/asr1/local/data.sh`
- ESPnet3: `egs3/librispeech_100/asr/src/create_dataset.py`
  (downloads/extracts into `<dataset_dir>/LibriSpeech/...`)

Your `create_dataset()` should be:

- deterministic
- safe to re-run (skip if already extracted)
- limited in scope (do “download + arrange”, not “feature extraction”)

Stage doc:

- [Create dataset](../stages/create-dataset.md)

#### Pattern B: manifest-based dataset

Generate TSV manifests under `data/<dataset>/manifest/*.tsv` and let your
`src/dataset.py` read those manifests.

mini_an4 is an example:

- recipe: `egs3/mini_an4/asr/src/create_dataset.py`
- outputs: `data/mini_an4/manifest/{train_dev,train_nodev,test}.tsv`

This pattern is also convenient for CI/integration tests because the manifests
are explicit and small.

#### Case study: LibriSpeech 100h migration notes (egs2 -> egs3)

ESPnet2 LibriSpeech 100h data prep lives in:

- `egs2/librispeech_100/asr1/local/data.sh`

The key steps and a reasonable ESPnet3 mapping:

1. **Download/extract**
   - ESPnet2: loop over `dev-clean`, `test-clean`, `dev-other`, `test-other`, `train-clean-100`
   - ESPnet3: `create_dataset(dataset_dir=...)` downloads/extracts into `<dataset_dir>/LibriSpeech/...`
     (see `egs3/librispeech_100/asr/src/create_dataset.py`)

2. **Kaldi data-dir generation**
   - ESPnet2: `local/data_prep.sh` writes `data/<split>/{wav.scp,text,...}`
   - ESPnet3: often *skip this entirely* and read the native corpus layout:
     `src/dataset.py` scans `*.trans.txt` and reads `<utt_id>.flac` directly.

3. **Combine dev sets**
   - ESPnet2: `utils/combine_data.sh ... data/dev_clean data/dev_other`
   - ESPnet3: model it explicitly in your DataOrganizer:
     keep `dev-clean` and `dev-other` as separate validation sets, or merge them
     in a recipe-specific dataset wrapper if you really need a single `valid`.

4. **External text**
   - ESPnet2: downloads LibriSpeech LM text and formats it
   - ESPnet3: implement a tokenizer text builder function and point
     `train_config.tokenizer.text_builder.func` to it.
     Start simple by using the training transcripts first (see
     `egs3/librispeech_100/asr/src/tokenizer.py`), then add external text if needed.

For the “where does this connect” view:

- dataset creation hook: `espnet3/systems/asr/system.py` (`ASRSystem.create_dataset()`)
- stage behavior: [Create dataset](../stages/create-dataset.md)
- dataset wiring: [DataOrganizer and dataset pipeline](./components/data-organizer.md)

### 4) Implement the dataset loader: `src/dataset.py`

In ESPnet3, the training pipeline expects your dataset to return dict-like
items that include (at least) the model inputs and labels you want to train on.

Common conventions for ASR:

- `speech`: waveform (float32 array)
- `text`: transcript
- optionally a stable id key (`utt_id`, etc.) for debugging/logging

Concrete example (LibriSpeech native layout):

- `egs3/librispeech_100/asr/src/dataset.py`

Concrete example (manifest-based):

- `egs3/mini_an4/asr/src/dataset.py`

Background doc (how datasets are wired into training):

- [DataOrganizer and dataset pipeline](./components/data-organizer.md)

### 5) Port tokenizer training: external text -> `src/tokenizer.py`

If your ESPnet2 recipe trains SentencePiece/BPE using:

- `--bpe_train_text ...`
- `--lm_train_text ...`
- “external text” in `local/data.sh` (e.g., LibriSpeech LM text download)

Then implement a text builder function for ESPnet3 and point your train config
to it.

LibriSpeech example:

- `egs3/librispeech_100/asr/src/tokenizer.py` provides `gather_training_text(...)`

The System calls the builder via `train_config.tokenizer.text_builder.func` and
feeds it into the SentencePiece trainer (ASRSystem behavior).

Related docs:

- [Train stage](../stages/train.md)
- [Train configuration reference](../config/train_config.md)

### 6) Create configs: `conf/train.yaml`, `conf/infer.yaml`, `conf/metric.yaml`, ...

This is where most migration time goes. The workflow that scales is:

1. start from a working ESPnet3 config (TEMPLATE or mini_an4)
2. copy relevant hyperparameters from ESPnet2 configs
3. keep a “single source of truth” for shared variables (paths, exp_tag, tokenizer)

Config references (don’t memorize; link and consult when editing):

- [Train configuration reference](../config/train_config.md)
- [Inference configuration reference](../config/infer_config.md)
- [Metric configuration reference](../config/metric_config.md)
- [Publish configuration reference](../config/publish_config.md)
- [Demo configuration reference](../config/demo_config.md)

Implementation references (when you want to see what keys are actually used):

- training entrypoint: `espnet3/systems/base/training.py`
- LightningModule config contract: `espnet3/components/modeling/lightning_module.py`
- inference entrypoint: `espnet3/systems/base/inference.py`
- metric entrypoint: `espnet3/systems/base/metric.py`


#### LibriSpeech 100h: what to copy from ESPnet2

From `egs2/librispeech_100/asr1/run.sh`:

```bash
--nbpe 5000
--feats_type raw
--speed_perturb_factors "0.9 1.0 1.1"
--max_wav_duration 30
```

Typical ESPnet3 mapping:

- `nbpe` -> `tokenizer.vocab_size` (+ tokenizer save_path)
- `feats_type raw` -> keep `speech` as waveform; configure frontend/preprocessor in model
- `speed_perturb_factors` -> `dataset.preprocessor.data_aug_effects` (see LibriSpeech tuning config)
- `max_wav_duration` -> usually handled by dataset filtering (recipe-specific; start without it)

#### Minimal example snippets (copy/paste starting points)

Train config skeleton (start from TEMPLATE, then edit):

```yaml
num_device: 1
num_nodes: 1
task: espnet3.systems.asr.task.ASRTask

recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: your_exp_tag
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
dataset_dir: /path/to/dataset

create_dataset:
  func: src.create_dataset.create_dataset
  dataset_dir: ${dataset_dir}

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train: [...]
  valid: [...]
  preprocessor: ...

tokenizer:
  vocab_size: 5000
  model_type: bpe
  save_path: ${data_dir}/${tokenizer.model_type}_${tokenizer.vocab_size}
  text_builder:
    func: src.tokenizer.gather_training_text
    dataset_dir: ${dataset_dir}

dataloader: ...

optimizer:
  _target_: torch.optim.Adam
  lr: 0.002

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 15000

trainer: ...
best_model_criterion: ...
fit: {}
```

Inference config skeleton:

```yaml
num_device: 1
num_nodes: 1

recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: ${load_yaml:conf/train.yaml,exp_tag}
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
inference_dir: ${exp_dir}/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test: [...]

parallel:
  env: local
  n_workers: 1

# Provider/Runner are required by `espnet3.systems.base.inference.infer()`.
provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider
  params: {}

runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/last.ckpt
  beam_size: 1
  ctc_weight: 0.3

input_key: speech
output_fn: src.inference.output_fn
```

Metric config skeleton (ASR WER/CER):

```yaml
recipe_dir: .
exp_tag: ${load_yaml:conf/train.yaml,exp_tag}
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test: [...]

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
  - metric:
      _target_: espnet3.systems.asr.metrics.cer.CER
```

Notes:

- For complete field lists, use the config reference docs linked above.
- For metric implementation details, see [ESPnet3 Metrics](./components/metrics.md).
  Also see the parallel/runtime docs:
  [Provider / Runner](./parallel/provider_runner.md), [Parallel](./parallel.md).

### 7) Wire `run.py` (concept + minimum working example)

Most recipes keep `run.py` tiny and reuse the shared stage driver logic.

Example (ASR, minimal):

```python
from egs3.TEMPLATE.asr.run import ALL_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(stages=ALL_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=ALL_STAGES)
    main(args=args, system_cls=ASRSystem, stages=stages_to_run)
```

Conceptually, the stage driver does:

```python
train_cfg = load_config_with_defaults(args.train_config)
infer_cfg = load_config_with_defaults(args.infer_config)
metric_cfg = load_config_with_defaults(args.metric_config)

system = ASRSystem(train_config=train_cfg, infer_config=infer_cfg, metric_config=metric_cfg)
run_stages(system, stages_to_run)
```

See the template implementation:

- `egs3/TEMPLATE/asr/run.py`
- `espnet3/utils/stages_utils.py`

If you copied an older template and hit import/name mismatches, check these
common renames:

- `optim` -> `optimizer` (train config key)
- `espnet3.utils.stages` -> `espnet3.utils.stages_utils` (module name)

### 8) Run the pipeline (recommended order)

Start with `--dry_run` once to validate CLI/config wiring.

```bash
python run.py --stages all --train_config conf/train.yaml --dry_run
```

Then run the typical ASR flow:

```bash
python run.py --stages create_dataset train_tokenizer collect_stats train \\
  --train_config conf/train.yaml

python run.py --stages infer --infer_config conf/infer.yaml

python run.py --stages metric --metric_config conf/metric.yaml
```

Publish and demo checks:

```bash
python run.py --stages pack_model upload_model \\
  --train_config conf/train.yaml \\
  --publish_config conf/publish.yaml

python run.py --stages pack_demo upload_demo --demo_config conf/demo.yaml
```

Stage docs (for “what does this stage do?”):

- [Create dataset](../stages/create-dataset.md)
- [Collect stats](../stages/collect-stats.md)
- [Train](../stages/train.md)
- [Inference](../stages/inference.md)
- [Metric](../stages/metrics.md)
- [Publish](../stages/publish.md)
- [Demo](../stages/demo.md)

## Tests (required for recipe PRs)

### Unit tests for new `espnet3/` code

If you add or modify core implementation under `espnet3/`, add a unit test under
`test/espnet3/`.

Browse existing tests:

- `test/espnet3/systems/`
- `test/espnet3/components/`
- `test/espnet3/demo/`

### Integration test: always include mini_an4

Add at least one integration test that runs a small pipeline on `egs3/mini_an4`.
This should be lightweight (CI-friendly) and should validate that:

- stages run end-to-end (at least `train` + `infer` + `metric`)
- expected artifact files are created (config, checkpoint, `.scp`, metrics.json)

Tip: mini_an4 already contains configs you can point to:

- `egs3/mini_an4/asr/conf/train_asr_transformer_debug.yaml`
- `egs3/mini_an4/asr/conf/infer.yaml`
- `egs3/mini_an4/asr/conf/measure.yaml` (legacy name; prefer `metric.yaml`, see note above)

## Docstrings (required for new recipe scripts)

Write detailed docstrings for new functions/classes. Each docstring should
include:

- Args
- Returns
- Raises
- Notes
- Examples
- Detailed description of the function/class

Sample prompt for a coding agent:

```text
Please create detailed docstrings for the `...` directory. Each docstring should include
Args, Returns, Raises, Notes, Examples, and detailed description of the function/class.
```

## PR checklist (before opening a recipe PR)

- [ ] Recipe runs end-to-end locally with expected outputs.
- [ ] `conf/train.yaml` exists and is the “shared values” anchor (other configs may reference it).
- [ ] New core code under `espnet3/` has unit tests under `test/espnet3/`.
- [ ] An integration test exists (mini_an4).
- [ ] `demo` stage works (at least `pack_demo` produces a runnable demo directory).
- [ ] Docstrings are complete and helpful.

## Useful “where to look” links

- [ESPnet3 homepage](../../espnet3_tutorial.md)
- [Getting started](../get_started.md)
- [Recipe directory layout](../recipe_directory.md)
- [Systems](./systems.md)
- [Components](./components.md)
- [Demo (developer notes)](./demo.md)

## Notes / upstream reference

If you want a reference tree that includes bugfixes and review feedback, you may
want to start from:

```bash
git clone https://github.com/Masao-Someki/espnet.git -b espnet3/recipe/ls_asr100_2
```

This page should be updated as ESPnet3 changes (bugfixes, PR reviews, and stage
API adjustments).
