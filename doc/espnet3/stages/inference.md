---
title: 📘 ESPnet3 Inference Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Inference Stage

This document explains the **inference stage** in ESPnet3, implemented in:

* `espnet3.systems.base.inference.inference`
* `espnet3.systems.base.inference_provider.InferenceProvider`
* `espnet3.systems.base.inference_runner.InferenceRunner`

Inference writes one or more `.scp` files (e.g., `hyp.scp`) that the
metrics stage later consumes. See `metrics.md` for metric computation.

## Quick usage

### Run

```bash
python run.py --stages infer --infer_config conf/infer.yaml
```

### Configure (in `infer.yaml`)

Keep the core settings in `infer.yaml`. For the full list, see
[Inference configuration](../config/infer_config.md).

| Config section | Description |
| -------------- | ----------- |
| `model` | Hydra target for the inference model (espnet2 or custom). Instantiated with a `device` argument. |
| `dataset` | Dataset organizer and test splits. The stage selects the test set named by `test_set`. |
| `parallel` | Parallel execution settings (e.g., local vs Dask, worker count). |
| `inference_dir` | Output location for `.scp` files under `inference_dir/<test_name>/`. |
| `input_key` | Which dataset field(s) to pass into the model. |
| `output_fn` | Import path to a function that formats runner outputs. |

See also:

- [Provider / Runner](../core/parallel/provider_runner.md)
- [Inference provider](../core/parallel/inference_provider.md)
- [Parallel execution](../core/parallel.md)

### Outputs

For each test set name in `dataset.test`, inference writes `.scp` files under:

```text
<inference_dir>/<test_name>/
```

The filenames are determined by:

- `output_keys` (if set), or
- the keys returned by `output_fn` for the first sample (excluding `idx_key`).

Each `.scp` file contains lines like:

```text
utt_id VALUE...
```

If `output_fn` returns a list for a given key (e.g., multiple hypotheses), each
entry is written to its own file: `<key>0.scp`, `<key>1.scp`, ...

## Developer Notes

### 🏃‍♂️ Inference with `InferenceRunner`

ESPnet3 inference is a Provider/Runner loop. `infer.yaml` provides two key
pieces:

- `input_key`: which field(s) to read from each dataset item and pass to the model
- `output_fn`: how to turn `model_output` into named outputs written as `.scp`

Conceptually, `espnet3.systems.base.inference.inference()` does something like:

```python
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner

provider = InferenceProvider(
    config,
    params={
        "input_key": config.input_key,
        "output_fn_path": config.output_fn,
    },
)

runner = InferenceRunner(
    provider=provider,
    idx_key=config.get("idx_key", "uttid"),
    hyp_key=config.get("output_keys", []),  # optional
)

results = runner(range(len(provider.build_dataset(config))))
```

A minimal `infer_config` for inference looks like:

```yaml
inference_dir: exp/asr_example/infer

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_train_config: exp/asr_example/config.yaml
  asr_model_file: exp/asr_example/last.ckpt

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test-clean
      dataset:
        _target_: ...
    - name: test-other
      dataset:
        _target_: ...

parallel:
  env: local
  n_workers: 1

input_key: speech
output_fn: src.infer.output_fn
```

For each test set name in `dataset.test`, `inference()` writes one `.scp` file
per output key under `inference_dir/<test_name>/` (e.g., `hyp.scp`, `wav.scp`, ...).

### `output_fn`: formatting model outputs into SCP fields

`output_fn` is required and is called from `InferenceRunner` as:

```python
output_fn(data=data, model_output=model_output, idx=idx)
```

It must return a dict that includes:

- `idx_key` (default: `uttid`) as a scalar identifier used for `.scp` lines
- output fields (strings, or list of strings for multi-output)

The default `InferenceRunner` also validates that required keys exist. In the
base entrypoint, `ref` is treated as a required key by default, so most recipes
return both `hyp` and `ref` from `output_fn`.

Minimal example (ASR-style):

```python
def output_fn(*, data, model_output, idx):
    # data is a dataset item dict (must contain your utt id field)
    # model_output is whatever your model returns for that item
    return {
        "uttid": data["uttid"],
        "hyp": model_output[0][0],  # e.g., Speech2Text output
    }
```

How it is used inside `InferenceRunner` (simplified dummy code):

```python
def forward(idx, *, dataset=None, model=None, input_key=None, output_fn_path=None, **_):
    data = dataset[idx]
    output_fn = load_output_fn(output_fn_path)
    model_output = model(data[input_key])
    return output_fn(data=data, model_output=model_output, idx=idx)
```

Notes:

- `InferenceRunner.forward` accepts either a single index or a list of indices.
- There is no `batch_forward` hook; batching is handled by passing lists into
  `forward` when `batch_size` is set.
- If `batch_size` is unset, `forward` receives a single dataset item and `idx`
  is a scalar. If `batch_size` is set (>= 1), `forward` receives a list of
  dataset items and `idx` is a list.

### Batched inference (`batch_size`)

If you set `batch_size` in `infer.yaml`, `InferenceRunner` chunks indices and
passes a list of indices into `forward`. The model is called once per batch
with list-valued inputs (one list per `input_key`), and `output_fn` receives
batched data and indices.

Conceptually:

```python
indices = [0, 1, 2, 3]
data_batch = [dataset[i] for i in indices]
inputs_dict = {"speech": [d["speech"] for d in data_batch]}

model_output = model(**inputs_dict)
out = output_fn(data=data_batch, model_output=model_output, idx=indices)
```

Minimal batched `__call__` example:

```python
class MyModel:
    def __call__(self, speech):
        # speech: list[...] with length == batch size
        return {"text": ["dummy" for _ in speech]}
```

`output_fn` that supports both single-item and batched calls:

```python
def output_fn(*, data, model_output, idx):
    # single-item
    if isinstance(data, dict):
        return {
            "uttid": data["uttid"],
            "hyp": model_output["text"],
            "ref": data.get("ref_text", ""),
        }

    # batched: data is a list[dict], idx is a list[int]
    hyps = model_output["text"]
    return [
        {
            "uttid": item["uttid"],
            "hyp": hyp,
            "ref": item.get("ref_text", ""),
        }
        for item, hyp in zip(data, hyps)
    ]
```

If your task produces audio hypotheses (e.g., TTS), write the audio files under
`<inference_dir>/<test_name>/` (or a subdirectory), and put the file paths in the
corresponding `hyp.scp` entries. Ensure the output directory exists before
writing SCPs so `metric()` can load them reliably.

Example: audio hypotheses written as file paths

If you generate audio files, `hyp.scp` typically stores the generated file path
per utterance:

```text
utt001 <inference_dir>/<test_name>/audio/utt001.wav
utt002 <inference_dir>/<test_name>/audio/utt002.wav
```

Example directory tree:

```text
<inference_dir>/
└── <test_name>/
    ├── hyp.scp
    └── audio/
        ├── utt001.wav
        └── utt002.wav
```

## 🧪 Using a custom model

The snippet above assumes the espnet2 `Speech2Text` interface. When you write
your **own** model or inference wrapper, you can either adapt your model to the
default runner or provide a custom runner.

#### Write your own InferenceRunner


If your model has a different interface (e.g., already returns `(hyp, ref)`), you
can subclass `BaseRunner` and change only the `forward` method:

```python
from espnet3.parallel.base_runner import BaseRunner


class MyInferenceRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        hyp, ref = model(data)  # your own API
        return {"idx": idx, "hyp": hyp, "ref": ref}
```

Then, in a custom `inference()` function or system subclass, construct this
runner instead of the default `InferenceRunner`. The rest of the pipeline
(`metric()`, metrics, etc.) can stay the same as long as you produce the `.scp`
keys that your `metric.yaml` expects (via `metrics[*].inputs`), such as
`hyp.scp` (and `ref.scp` if you choose to write references).
