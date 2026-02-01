---
title: 📘 ESPnet3 Inference & Metrics Framework
author:
  name: "Masao Someki"
date: 2025-11-26
---

This document explains the **infer + metric** evaluation flow in ESPnet3,
implemented in:

* `espnet3.systems.base.inference_provider.InferenceProvider` / `InferenceRunner` / `inference`
* `espnet3.systems.base.metric.metric` and `espnet3.components.metrics.abs_metric.AbsMetric`

It is designed for ASR-style evaluation but can be adapted to custom models as
long as you follow the expected interfaces.

## ✅ Who implements which part?

| Layer                        | You implement / configure                              | ESPnet3 handles                                      |
| ---------------------------- | ------------------------------------------------------ | --------------------------------------------------- |
| Model (espnet2 or custom)    | `config.model` Hydra target                            | Instantiation with a `device` argument              |
| Dataset / test splits        | `config.dataset` (usually `DataOrganizer`)             | Selecting the requested test set via `test_set`     |
| Decoding logic               | (Optional) custom `InferenceRunner.forward`            | Parallel execution via `BaseRunner`                 |
| Metrics                      | `AbsMetric` subclasses and `config.metrics` entries   | Loading SCPs and calling metrics via `metric()`    |
| Evaluation config YAML       | All of the above (model, dataset, parallel, metrics)   | Wiring into `BaseSystem.infer()` / `metric()`      |

## 🧠 Pipeline overview

At evaluation time, `BaseSystem` wires things as:

```text
BaseSystem.infer()
└── infer() -> espnet3.systems.base.inference.inference(infer_config)
    └── writes inference_dir/<test_name>/{ref,hyp}.scp

BaseSystem.metric()
└── metric() -> espnet3.systems.base.metric.metric(metric_config)
    └── reads those SCPs and computes metrics → writes metrics.json
```

In practice, these are run as two separate stages with two configs
(`infer.yaml` and `metric.yaml`). They typically share the same `dataset.test`
definitions, and metric reads the SCPs written by infer.

## 🏃‍♂️ Inference with `InferenceRunner`

The default ASR evaluation uses `espnet3.systems.base.inference`:

```python
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.inference_provider import InferenceProvider as BaseInferenceProvider


class InferenceProvider(BaseInferenceProvider):
    @staticmethod
    def build_dataset(config):
        organizer = instantiate(config.dataset)
        test_set = config.test_set
        return organizer.test[test_set]

    @staticmethod
    def build_model(config):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = instantiate(config.model, device=device)
        return model


class InferenceRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        speech = data["speech"]
        hyp = model(speech)[0][0]
        ref = model.tokenizer.tokens2text(
            model.converter.ids2tokens(data["text"])
        )
        return {"idx": idx, "hyp": hyp, "ref": ref}


def inference(config: DictConfig):
    set_parallel(config.parallel)
    test_sets = [t.name for t in config.dataset.test]
    for test_name in test_sets:
        config.test_set = test_name
        provider = InferenceProvider(config)
        runner = InferenceRunner(provider=provider, async_mode=False)
        out = runner(range(len(provider.build_dataset(config))))
        ...
```

### 🔧 Config fields used during inference

A minimal `infer_config` for inference looks like:

```yaml
inference_dir: exp/asr_example/infer

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_model_file: path/to/model.ckpt

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
```

For each test set name in `dataset.test`, `inference()` writes:

```text
<inference_dir>/<test_name>/ref.scp
<inference_dir>/<test_name>/hyp.scp
```

Each `.scp` file contains lines like:

```text
utt_id REF TEXT ...
utt_id HYP TEXT ...
```

## 🧪 Using a custom model

The snippet above assumes the espnet2 `Speech2Text` interface. When you write
your **own** model or inference wrapper, there are two main options.

### Option 1: Keep the default `InferenceRunner`

Make your model behave like `Speech2Text`:

- Accept `device` as an argument in the Hydra-instantiated class.
- Implement `__call__(self, speech)` so that it returns a list where
  `model(speech)[0][0]` is a string hypothesis.
- Expose `tokenizer` and `converter` attributes if you also want the default
  reference reconstruction.

Example config:

```yaml
model:
  _target_: my_project.inference.CustomSpeech2Text
  asr_model_file: path/to/model.ckpt
```

This way you can reuse the built-in `InferenceRunner` unchanged.

### Option 2: Write your own `InferenceRunner`

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
(`metric()`, metrics, etc.) can stay the same as long as you still produce
`ref.scp` and `hyp.scp`.

## 📏 Metrics with `AbsMetric`

Metrics are implemented in `espnet3.systems.base.metric.metric` and use the
`AbsMetric` base class from `espnet3.components.metrics.abs_metric`:

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class AbsMetric(ABC):
    @abstractmethod
    def __call__(
        self, data: Dict[str, List[str]], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        ...
```

`metric()` reads the SCP files, aligns utterance IDs, and then calls your metric:

```python
from hydra.utils import instantiate
from espnet3.components.metrics.abs_metric import AbsMetric
from espnet3.utils.scp_utils import load_scp_fields


def metric(config):
    test_sets = [t.name for t in config.dataset.test]
    for metric_cfg in config.metrics:
        metric = instantiate(metric_cfg.metric)
        inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
        data = load_scp_fields(
            inference_dir=Path(config.inference_dir),
            test_name=test_name,
            inputs=inputs,
            file_suffix=".scp",
        )
        result = metric(data, test_name, Path(config.inference_dir))
        ...
```

### 🔹 Metric config (`metrics` section)

Each metric entry has three important fields:

- `metric`: Hydra target for your `AbsMetric` subclass.
- `inputs`: which SCP files to read and how to name them.
- `apply_to`: list of test set names to evaluate on.

Example:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
    inputs:
      ref: ref      # reads ref.scp    → data["ref"]
      hyp: hyp      # reads hyp.scp    → data["hyp"]
    apply_to: [test-clean, test-other]
```

### 🔹 Implementing a custom metric

```python
from espnet3.components.metrics.abs_metric import AbsMetric
import jiwer


class WER(AbsMetric):
    def __call__(self, data, test_name, output_dir):
        refs = data["ref"]
        hyps = data["hyp"]
        wer = jiwer.wer(refs, hyps)
        return {"WER": round(wer * 100, 2)}
```

As long as the returned dict is JSON-serializable, `metric()` will write it into
`<inference_dir>/metrics.json`.

## 🧰 Putting it together (`infer.yaml` + `metric.yaml`)

A simplified end-to-end evaluation setup uses two stage configs.

`infer.yaml`:

```yaml
inference_dir: exp/asr_example/infer

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_model_file: path/to/model.ckpt

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
```

`metric.yaml`:

```yaml
inference_dir: exp/asr_example/infer

dataset:
  _target_: ${load_yaml:conf/infer.yaml,dataset._target_}
  test: ${load_yaml:conf/infer.yaml,dataset.test}

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
    inputs:
      ref: ref
      hyp: hyp
    apply_to: [test-clean, test-other]
```

Then run:

1. `infer` → `inference(infer_config)` to create SCP files.
2. `metric` → `metric(metric_config)` to compute metrics and write `metrics.json`.

## ✅ Summary

| Stage   | What you must provide                                  | Files produced / consumed                      |
| ------- | ------------------------------------------------------ | ---------------------------------------------- |
| Infer   | Model config (espnet2 or custom), dataset, parallel    | `<inference_dir>/<test>/ref.scp`, `hyp.scp`       |
| Metric | `AbsMetric` subclasses and `metrics` config           | `metrics.json` with per-metric, per-test scores |

When you bring a custom model, the main work is to either:

- mimic the `Speech2Text` interface so that the default `InferenceRunner`
  continues to work, or
- write a small `BaseRunner` subclass that returns `{"idx", "hyp", "ref"}` so
  that the metrics side can stay unchanged.
