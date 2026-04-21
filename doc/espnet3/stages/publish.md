---
title: ESPnet3 Publication Stages
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Publication Stages

Current publication flow is centered on:

- `pack_model`
- `upload_model`

Implementation:

- `espnet3.utils.publish_utils.pack_model`
- `espnet3.utils.publish_utils.upload_model`

## Quick usage

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

## `pack_model`

`pack_model` creates a publishable bundle directory, usually:

```text
<exp_dir>/model_pack
```

### Strategy

`pack_model.strategy` supports:

- `auto`
- `espnet2`
- `espnet3`

Current behavior:

- if `training_config.task` is set, packing defaults to the espnet2 path
- otherwise it uses the espnet3 path

### What gets bundled

Current ESPnet3 packing copies recipe assets such as:

- `conf/`
- `src/`
- `run.py`
- `pixi.toml`
- `pixi.lock`
- `.python-version`

and usually includes the experiment directory and, when enabled, the recipe
`data_dir`.

In practice the packed tree often looks like:

```text
model_pack/
  conf/
  src/
  exp/
  data/
  run.py
  meta.yaml
  README.md
  scores.json
```

`data/` is included when `include_data_dir: true` and the configured `data_dir`
exists.

### Additional file controls

`publication.yaml` can also define:

- `include`
- `extra`
- `exclude`
- `files`
- `yaml_files`

These allow adding or explicitly registering artifacts in `meta.yaml`.

## `meta.yaml`

`pack_model` writes `meta.yaml` into the bundle root. That metadata is later
used by tools such as `InferenceSession`.

## `InferenceSession`

Direct packaged-model inference is provided by:

- `espnet3.publication.InferenceSession`

Typical use:

```python
from espnet3.publication import InferenceSession

session = InferenceSession.from_pretrained(
    "espnet/your-model-tag",
    trust_user_code=True,
)
result = session(audio_array)
```

Important behavior:

- it can load `conf/inference.yaml` from the packed bundle
- it can use bundle metadata from `meta.yaml`
- it can enable bundled recipe code such as `src/` when
  `trust_user_code=True`

This is the main reason current espnet3 packing includes recipe configs and
recipe-local user code.

## `upload_model`

`upload_model` uploads the packed directory to Hugging Face.

Required field:

- `publication_config.upload_model.hf_repo`

The packed directory must already exist before upload runs.

## Related pages

- [Publication configuration](../config/publish_config.md)
- [Inference stage](./inference.md)
