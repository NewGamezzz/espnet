---
title: 📘 ESPnet3 Inference Stage
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Stage

The current inference entrypoint is:

- `espnet3.systems.base.inference.infer`

The default runner stack is:

- `espnet3.systems.base.inference_provider.InferenceProvider`
- `espnet3.systems.base.inference_runner.InferenceRunner`

## Quick usage

### Run

```bash
python run.py --stages infer --inference_config conf/inference.yaml
```

### Configure (in `inference.yaml`)

Keep the core settings in `inference.yaml`. For the full list, see
[Inference configuration](../config/infer_config.md).

| Config section | Description |
| --- | --- |
| `model` | Hydra target for the inference model |
| `dataset` | test-set definitions selected by the stage |
| `parallel` | local or distributed runner settings |
| `inference_dir` | root output location for SCP files |
| `input_key` | dataset field or fields passed into the model |
| `output_fn` | import path to the formatting function |

## Main outputs

Inference writes one directory per test set:

```text
<inference_dir>/
  <test_name>/
    hyp.scp
    ...
```

Each SCP line is:

```text
utt_id value
```

If an output field is non-scalar, ESPnet3 writes an artifact under the test-set
directory and stores the artifact path in the SCP file.

### Artifact outputs

`output_fn` can return:

- scalar values such as `str`, `int`, `float`, `bool`
- non-scalar values handled through `output_artifacts`

Scalar values are written directly into SCP files.

Non-scalar values are written as artifacts under:

```text
<inference_dir>/<test_name>/<field_name>/
```

and the SCP file stores the written path.

Built-in artifact behavior:

| Value type | Default artifact type | Saved as |
| --- | --- | --- |
| `dict` | `json` | `.json` |
| `numpy.ndarray` | `npy` | `.npy` |
| CPU `torch.Tensor` | `npy` | `.npy` |
| other Python object | `pickle` | `.pkl` |

Config can also force a type such as `wav`.

### WAV example

If `output_fn` returns:

```python
{
    "utt_id": "utt1",
    "audio": wav_numpy,
}
```

and `inference.yaml` contains:

```yaml
output_artifacts:
  audio:
    type: wav
    sample_rate: 16000
```

then inference writes:

```text
<inference_dir>/
  <test_name>/
    audio.scp
    audio/
      utt1.wav
      utt2.wav
```

and `audio.scp` stores the generated `.wav` paths.

### Custom artifact writer

If you want to save a custom type such as PNG, add a writer function and point
to it from config.

Example config:

```yaml
output_artifacts:
  image:
    writer:
      _target_: src.inference.write_png_artifact
```

Example function:

```python
from pathlib import Path


def write_png_artifact(*, value, output_path):
    path = Path(output_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path)
    return path
```

The writer must return the written path. That path is stored in the SCP file.

### Output directory layout

For each test-set name in `dataset.test`, inference writes:

```text
<inference_dir>/<test_name>/
```

The filenames are determined by:

- `output_keys` when it is set
- otherwise the keys returned by `output_fn` for the first sample, excluding
  `idx_key`

### Conceptual provider and runner flow

Inference is a Provider/Runner loop. Conceptually:

```python
provider = InferenceProvider(config)
runner = InferenceRunner(provider=provider, async_mode=False)
results = runner(range(len(provider.build_dataset(config))))
```

The provider is responsible for:

- building the dataset for the active test set
- instantiating the model
- exposing config-derived runtime parameters

The runner is responsible for:

- pulling one sample or one batch from the dataset
- calling the model with the configured `input_key`
- normalizing the result through `output_fn`
- returning values that can be written into SCP files

## Experiment naming and `exp_tag`

If `training_config` is loaded in the same `run.py` call, inference inherits:

- `exp_tag`
- `exp_dir`

If inference runs alone, it uses its own `inference.yaml` values.

### Example: inherited from training

`training.yaml`:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

`inference.yaml`:

```yaml
exp_tag: inference_beam5
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

Run:

```bash
python run.py \
  --stages train infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```

Final values:

- `exp_tag = training_branchformer`
- `exp_dir = ${recipe_dir}/exp/training_branchformer`
- `inference_dir = ${recipe_dir}/exp/training_branchformer/inference`

### Example: not inherited

`inference.yaml`:

```yaml
exp_tag: inference_beam5
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

Run:

```bash
python run.py \
  --stages infer \
  --inference_config conf/inference.yaml
```

Final values:

- `exp_tag = inference_beam5`
- `exp_dir = ${recipe_dir}/exp/inference_beam5`
- `inference_dir = ${recipe_dir}/exp/inference_beam5/inference`

## `output_fn`

`output_fn` is called right after the model returns.

The order is:

1. load one sample or one batch from the dataset
2. call the model with `input_key`
3. call `output_fn`
4. write scalar values to SCP files or write artifacts to disk

If provided, `output_fn` is called as:

```python
output_fn(data=data, model_output=model_output, idx=idx)
```

It should return a dict for a single sample, or a list of dicts for batched
inference.

Typical output:

```python
{
    "utt_id": "utt1",
    "hyp": "hello world",
}
```

The base runner accepts either a single index or a list of indices. That is why
`output_fn` must be able to handle:

- a single sample plus scalar `idx`
- or batched input where `data` is a list and `idx` is a list

Minimal single-sample example:

```python
def build_output(*, data, model_output, idx):
    return {
        "utt_id": data["uttid"],
        "hyp": model_output["text"],
        "ref": data.get("text", ""),
    }
```

## Batched inference

If `batch_size` is set, `InferenceRunner.forward()` receives a list of indices
and passes list-valued inputs to the model.

If your model or `output_fn` does not support batched list inputs, leave
`batch_size` unset or `null`.

Minimal batched example:

```python
def build_output(*, data, model_output, idx):
    return [
        {
            "utt_id": item["uttid"],
            "hyp": hyp,
            "ref": item.get("text", ""),
        }
        for item, hyp in zip(data, model_output["text"])
    ]
```

This is why the document examples keep `output_fn` small: most recipe-specific
formatting problems are easier to solve there than by replacing the whole stage.

## Dataset naming

`dataset.test[*].name` becomes:

- the selected test-set key
- the subdirectory name under `inference_dir`

This same test-set name is later reused by `measure()`.

## Using a custom model

Two common paths:

1. keep `InferenceRunner` and replace only `model` and `output_fn`
2. replace `InferenceRunner` when the normal flow is not enough

### Example: custom decoding algorithm

This is the common case:

- you want to keep the same dataset
- you want to keep the same SCP writing path
- but you want your own decoding algorithm

In that case, keep the default runner and replace only `model` and `output_fn`.

Example `inference.yaml`:

```yaml
dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test

model:
  _target_: src.inference.MyGreedyDecoder
  checkpoint_path: ${exp_dir}/last.ckpt
  beam_size: 1

input_key: speech
output_fn: src.inference.build_output

provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider
runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
```

Example `src/inference.py`:

```python
from pathlib import Path

import torch


class MyGreedyDecoder:
    def __init__(self, checkpoint_path, beam_size=1):
        self.checkpoint_path = Path(checkpoint_path)
        self.beam_size = beam_size
        self.model = self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        model.eval()
        return model

    def __call__(self, speech):
        tokens = self.model.decode(speech, beam_size=self.beam_size)
        text = self.model.tokenizer.decode(tokens)
        return {"text": text, "tokens": tokens}


def build_output(*, data, model_output, idx):
    return {
        "utt_id": data.get("uttid", str(idx)),
        "hyp": model_output["text"],
        "token_ids": " ".join(str(v) for v in model_output["tokens"]),
        "ref": data.get("text", ""),
    }
```

The runtime order is:

1. `InferenceRunner` loads one sample from the dataset
2. it calls `model(**inputs)`
3. it calls `build_output(...)`
4. it writes `hyp.scp`, `token_ids.scp`, and `ref.scp`

### When to replace `InferenceRunner`

Replace the runner only when `model -> output_fn -> SCP` is not enough.

Examples:

- streaming decode with internal state
- multi-step search with custom batching
- non-standard output validation

Minimal custom runner example:

```python
from espnet3.systems.base.inference_runner import InferenceRunner


class MyInferenceRunner(InferenceRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        model_output = model.decode_stream(data["speech"])
        return {
            "utt_id": data.get("uttid", str(idx)),
            "hyp": model_output["text"],
            "ref": data.get("text", ""),
        }
```

Config:

```yaml
runner:
  _target_: src.inference.MyInferenceRunner
```

Use this path only when `output_fn` is not enough.

## Related pages

- [Inference configuration](../config/infer_config.md)
- [Measure stage](./measure.md)
- [Provider / runner](../core/parallel/provider_runner.md)
