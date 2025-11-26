---
title: 📘 ESPnet3 Inference & Evaluation Framework
author:
  name: "Masao Someki"
date: 2024-07-01
---

This document provides a deep explanation of the inference and scoring pipeline in ESPnet3, centered around the following components:

* `InferenceRunner`: General-purpose inference controller
* `ASRInferenceRunner`: Concrete subclass for ASR models
* `AbsMetrics` & `ScoreRunner`: Abstract metric interface and evaluation controller
* YAML-driven configuration for reproducible experiments

---

## 🧠 System Overview

```text
evaluate.py
├── loads evaluate.yaml
├── initializes ASRInferenceRunner (InferenceRunner subclass)
├── runs decoding per test set (serial or parallel)
└── runs scoring using ScoreRunner
```

---

## 🏃‍♂️ `InferenceRunner` – Generic Inference Framework

### Purpose

A flexible, extensible class for managing **test-time inference**, supporting:

* Parallel decoding (via Dask)
* Streaming mode
* Kaldi-style output (SCP format)
* Resumable decoding
* Hooks for customization (`pre_inference`, `inference_body`, `post_inference`)

---

### 🔧 Key Hooks to Override

| Method               | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `initialize_model`   | Instantiate model using Hydra configuration                                  |
| `initialize_dataset` | Load dataset using DataOrganizer config, with transform/preprocessor applied |
| `pre_inference`      | Optional preprocessing step (e.g., tokenization, padding)                    |
| `inference_body`     | Core inference logic – must be implemented by subclass                       |
| `post_inference`     | Merge streamed outputs or perform any post-processing                        |

---

### 🗂 Sample I/O Format

Model must return a dictionary per sample like:

```python
{
    "hypothesis": {"type": "text", "value": "hello world"},
    "rtf": {"type": "text", "value": "0.3142"},
    "ref": {"type": "text", "value": "hello world"},  # optional
    "weight": {"type": "image", "value": np.ndarray}, # optional
    "wav": {"type": "audio", "value": np.ndarray}     # optional
}
```

These entries are written to:

* `decode_dir/text.scp`
* `decode_dir/data/audio/utt.flac`
* etc.

---

### ⚡ Inference Execution Modes

| Mode         | Trigger                      | Description                       |
| ------------ | ---------------------------- | --------------------------------- |
| **Serial**   | `parallel: null`             | One-by-one decoding               |
| **Parallel** | `parallel: {env: local_gpu}` | Sample-level parallelism via Dask |
| **Async**    | `parallel + async_mode=True` | Chunk-level async decoding        |

---

### 🌊 Streaming Inference

Streaming is enabled with `stream: true`.
Audio or text input is read chunk-by-chunk, and the inference loop iterates over them.

```python
for chunk in sample["stream"]:
    output = model(chunk)
    ...
```

The merged result is handled in `post_inference()`.

---

### 🧪 Example: ASRInferenceRunner

```python
class ASRInferenceRunner(InferenceRunner, nn.Module):
    def inference_body(self, model, sample):
        speech = sample["speech"]
        results = model(speech)
        hyp_text = results[0][0] if results else ""
        ...
        return {
            "hypothesis": {"type": "text", "value": hyp_text},
            "rtf": {"type": "text", "value": str(round(rtf, 4))},
            "ref": {"type": "text", "value": sample["text"]},
        }
```

---

## 📏 `AbsMetrics` & `ScoreRunner` – Evaluation Framework

### 🔹 `AbsMetrics`

Abstract base class for scoring. Implements one method:

```python
class AbsMetrics(ABC):
    def __call__(self, data: Dict[str, List[str]], test_name: str, decode_dir: Path) -> Dict[str, float]
```

The format of `data` is `Dict[str, List[str]]`, which is something like:
```python
{
  "hypothesis": ["hyp1", "hyp2", ...],
  "ref": ["ref1", "ref2", ...]
}
```

### 🔹 Example: WER

```python
class WER(AbsMetrics):
    def __call__(self, data, test_name, decode_dir):
        refs = [self.clean(x) for x in data["ref"]]
        hyps = [self.clean(x) for x in data["hypothesis"]]
        score = jiwer.wer(refs, hyps)
        return {"WER": round(score * 100, 2)}
```

Text is optionally cleaned using `TextCleaner`.

---

### 🧪 `ScoreRunner`

Reads decoded `.scp` outputs (e.g., `ref.scp`, `hypothesis.scp`) and applies metrics defined in `evaluate.yaml`.

```yaml
metrics:
  - _target_: ...WER
    inputs: [ref, hypothesis]
    apply_to: [test-clean, test-other]
```

Each metric gets its own section in the summary output.

---

## 🧰 Configuration (evaluate.yaml)

```yaml
model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_model_file: path/to/model.ckpt

dataset:
  _target_: espnet3.data.DataOrganizer
  test:
    - name: test-clean
      dataset:
        _target_: ...
    - name: test-other
      dataset:
        _target_: ...

parallel:
  env: local_gpu
  n_workers: 4

metrics:
  - _target_: egs3.librispeech_100.asr1.score.wer.WER
    inputs: [ref, hypothesis]
```

---

## 🧪 One-Sample Debugging

```bash
python evaluate.py --stage decode --debug_sample
```

Useful for verifying model and data before full decoding.

---

## ✅ Summary

| Feature                | Supported? | Notes                                    |
| ---------------------- | ---------- | ---------------------------------------- |
| Serial inference       | ✅          | Default mode                             |
| Parallel Dask decoding | ✅          | With `parallel` config                   |
| Streaming inference    | ✅          | With `stream: true`                      |
| Kaldi-style outputs    | ✅          | `.scp` + file-based outputs (text/audio) |
| ASR evaluation (WER)   | ✅          | Plug-in based metric system              |
| Resumability           | 🟡         | Partial support via `.scp` checks        |
| RTF & memory tracking  | ✅          | Via `psutil` and timing metrics          |
