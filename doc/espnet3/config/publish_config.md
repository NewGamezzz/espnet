---
title: ESPnet3 Publication Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Publication Configuration

This page describes the current `publication.yaml` used by:

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

## Minimum required keys

Typical publication runs need:

- `pack_model`
- `upload_model.hf_repo` for upload

`pack_model` also needs `training_config`.

Optional but useful:

- `inference_config`
- `metrics_config`

These let the bundle keep `conf/inference.yaml`, `conf/metrics.yaml`, and
`scores.json`.

## Config sections overview

| Section | Description |
| --- | --- |
| `pack_model.strategy` | choose `auto`, `espnet2`, or `espnet3` packing |
| `pack_model.out_dir` | output bundle directory |
| `pack_model.decode_dir` | directory searched for `scores.json` |
| `pack_model.readme_template`, `readme_context` | README template and extra template values |
| `pack_model.include`, `extra`, `exclude` | extra copy and exclusion control |
| `pack_model.include_data_dir` | include `training_config.data_dir` in the bundle |
| `pack_model.files`, `yaml_files` | explicit metadata entries copied into the bundle |
| `pack_model.espnet2` | espnet2-specific packing spec |
| `upload_model` | Hugging Face upload settings |

## Default values

| Key | Default value |
| --- | --- |
| `pack_model.strategy` | `auto` |
| `pack_model.out_dir` | `exp/model_pack` |
| `pack_model.decode_dir` | `${exp_dir}/decode` |
| `pack_model.include` | `[]` |
| `pack_model.extra` | recipe-dependent |
| `pack_model.include_data_dir` | `false` in TEMPLATE |
| `pack_model.exclude` | `['**/*.log', '**/tensorboard/**', '**/wandb/**']` |
| `pack_model.files` | `{}` |
| `pack_model.yaml_files` | `{}` |
| `pack_model.espnet2.option` | `[]` |

## Typical usage

### Pack and upload in one run

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

This is the common case.

### Upload only

If the bundle already exists, `upload_model` can run alone.

Example:

```yaml
pack_model:
  out_dir: exp/my_bundle

upload_model:
  hf_repo: yourname/your-model-repo
```

```bash
python run.py \
  --stages upload_model \
  --publication_config conf/publication.yaml
```

In this case, `pack_model.out_dir` must already exist.

## Minimal example

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

This is the smallest user override.

## `pack_model`

`pack_model` creates a publishable bundle directory.

It requires:

- `training_config`
- `publication_config.pack_model`

### `strategy`

| Tag | Description |
| --- | --- |
| `auto` | use `espnet2` when `training_config.task` is set, otherwise use `espnet3` |
| `espnet2` | force espnet2 packing |
| `espnet3` | force espnet3 packing |

`auto` is the normal setting.

### `out_dir`

`out_dir` is the final bundle directory.

Important behavior:

- if `out_dir` already exists, `pack_model` removes it first
- then it writes a fresh bundle

Typical ESPnet3 bundle:

```text
${exp_dir}/model_pack/
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
  src/
  exp/
  data/
  run.py
  pixi.toml
  pixi.lock
  .python-version
  files/
  yaml_files/
  meta.yaml
  README.md
  scores.json
```

Notes:

- `data/` is present only when `include_data_dir: true`
- `scores.json` is present only when one file is found
- `conf/*.yaml` depends on which configs are loaded or available
- `pixi.*` and `.python-version` are copied only when present in the recipe

### What gets copied by default

ESPnet3 packing copies recipe assets such as:

- `conf/`
- `src/`
- `run.py`
- `pixi.toml`
- `pixi.lock`
- `.python-version`

For the espnet3 path, it also copies:

- `${training_config.exp_dir}`

For ASR, default artifact registration also adds:

- `asr_train_config -> <exp_dir>/config.yaml`
- `asr_model_file -> <exp_dir>/last.ckpt`

These entries go into `meta.yaml`.

### `decode_dir`

`decode_dir` is searched for `scores.json`.

Priority:

1. `pack_model.decode_dir`
2. `inference_config.decode_dir` if `pack_model.decode_dir` is not set

If one `scores.json` is found, it is copied to:

```text
<out_dir>/scores.json
```

If more than one `scores.json` is found, packing fails. Point
`pack_model.decode_dir` to one decode directory in that case.

### `readme_template` and `readme_context`

`pack_model` writes `README.md` into the bundle root.

By default, ESPnet3 uses the built-in TEMPLATE README.

You can override the template:

```yaml
pack_model:
  readme_template: egs3/your_recipe/asr/src/readme.md
  readme_context:
    project_name: My Model
```

The rendered README can include:

- repo name
- recipe name
- system name
- training config dump
- results table from `scores.json`

### `include`, `extra`, `exclude`

| Key | Description |
| --- | --- |
| `include` | paths copied into the bundle before `extra` |
| `extra` | more paths copied into the bundle |
| `exclude` | patterns skipped while copying |

Example:

```yaml
pack_model:
  include:
    - docs/model_card_assets
  extra:
    - exp/tokenizer
  exclude:
    - "**/*.log"
    - "**/wandb/**"
```

Use these keys when you want to carry extra recipe files into the bundle.

### `include_data_dir`

If `true`, `training_config.data_dir` is copied into the bundle when it exists.

Example:

```yaml
pack_model:
  include_data_dir: true
```

This is useful when inference or post-processing still needs files under
`data/`.

### `files` and `yaml_files`

These keys copy explicit artifacts into the bundle and register them in
`meta.yaml`.

Example:

```yaml
pack_model:
  files:
    asr_model_file: ${exp_dir}/last.ckpt
    bpemodel: ${exp_dir}/tokenizer/bpe.model
  yaml_files:
    asr_train_config: ${exp_dir}/config.yaml
```

Typical output:

```text
<out_dir>/
  files/
    asr_model_file.pth
    bpemodel.model
  yaml_files/
    asr_train_config.yaml
  meta.yaml
```

`meta.yaml` stores the relative paths.

### `espnet2`

`pack_model.espnet2` is used only when espnet2 packing is selected.

Required key:

- `pack_model.espnet2.task`

Example:

```yaml
pack_model:
  strategy: espnet2
  espnet2:
    task: asr
    files:
      asr_model_file: ${exp_dir}/last.ckpt
    yaml_files:
      asr_train_config: ${exp_dir}/config.yaml
    option: []
```

Use this path when you want the classic espnet2 pack logic.

## `meta.yaml`

`pack_model` writes `meta.yaml` into the bundle root.

It stores:

- copied `files`
- copied `yaml_files`
- extra fields such as `user_code_paths`

This metadata is later used by packaged-model inference.

## `InferenceSession`

Published bundles are designed to work with:

- `espnet3.publication.InferenceSession`

Typical use:

```python
import soundfile as sf

from espnet3.publication import InferenceSession

session = InferenceSession.from_pretrained(
    "espnet/your-model-tag",
    trust_user_code=True,
)

audio, sample_rate = sf.read("sample.wav", dtype="float32")
result = session(audio)
print(result)
```

Current behavior:

- if the bundle has `conf/inference.yaml`, `InferenceSession` loads it
- if the bundle has `meta.yaml`, it uses the recorded artifact paths
- if the config references `src.*`, use `trust_user_code=True`

This is why current publication keeps:

- `conf/`
- `src/`
- `meta.yaml`

See [Publication stages](../stages/publish.md) for stage flow.

## `upload_model`

Minimal example:

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

Required key:

- `hf_repo`

Current behavior:

- uploads the directory at `pack_model.out_dir`
- fails if that directory does not exist
- requires `huggingface-cli`
- creates the repo first when needed

## Notes

- use `publication.yaml`, not `publish.yaml`
- use `--publication_config`, not `--publish_config`
- `pack_model` is a stage config block inside `publication.yaml`

## Related pages

- [Publication stages](../stages/publish.md)
- [Inference configuration](./infer_config.md)
- [Inference stage](../stages/inference.md)
