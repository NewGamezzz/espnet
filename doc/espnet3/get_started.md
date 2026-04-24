---
title: Getting Started with ESPnet3
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-04-23
---

# 🚀 Getting Started with ESPnet3

1. [Quick Start](#-1-quick-start-asr-example)
2. [Workflow](#2-workflow)
   - [Recipes](#recipes)
   - [Systems](#systems)
   - [Stages](#stages)
   - [Configs](#configs)
   - [Experiments](#experiments)
   - [Putting everything together](#putting-everything-together)
3. [Additional ESPnet3 Documentation](#-3-additional-espnet3-documentation)
4. [Reference Snippets](#-4-reference-snippets)

- - -

# ⚡ 1. Quick Start (ASR Example)

## 📦 1.1. Install ESPnet3

ESPnet3 is distributed under the same pip package name as previous versions: `espnet`.
For more installation options, see [ESPnet3 Installation](./install.md).

```bash
pip install espnet
```

### Install from source (recommended for development)

```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools

# Recommended: setup_uv.sh
# Installs pixi + uv and sets up dependencies faster than the old conda flow.
. setup_uv.sh
```

## ⬇️ 1.2. Install system-specific dependencies

ESPnet3 introduces the concept of a system such as ASR, TTS, ST, or ENH.
Different systems may require different optional dependencies.

Install system extras with:

```bash
pip install "espnet[asr]"
```

Other examples:

```bash
pip install "espnet[tts]"
pip install "espnet[st]"
pip install "espnet[enh]"
```

If you installed from a cloned repository:

```bash
pip install -e ".[asr]"
# or using uv:
uv pip install -e ".[asr]"
```

For a full list of options, check the [installation page](./install.md)

## 🧪 1.3. Run a recipe

### Option 1: In Python
ESPnet3 recipes are importable, which makes it possible to build a recipe-driven pipeline in your own Python code.
This mode is useful for programmatic pipelines, notebooks, and MLOps workflows.

A minimal import-based execution example is:

```python
from argparse import Namespace
from pathlib import Path

from egs3.TEMPLATE.asr.run import main
from espnet3.systems.asr.system import ASRSystem

stages = ["create_dataset", "collect_stats", "train", "infer", "measure"]
args = Namespace(
    stages=stages,
    training_config=Path("/path/to/conf/training.yaml"),
    inference_config=Path("/path/to/conf/inference.yaml"),
    metrics_config=Path("/path/to/conf/metrics.yaml"),
    publication_config=None,
    demo_config=None,
    dry_run=False,
    write_requirements=False,
)

main(args=args, system_cls=ASRSystem, stages=stages)
```

### Option 2: From the Command Line

If you cloned the repository from source, you can run a recipe from within the `egs3/` directory.
All configs and helper scripts are located there.

Example: LibriSpeech 100h ASR

```bash
cd egs3/librispeech_100/asr
python run.py \
  --stages create_dataset collect_stats train infer measure \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

# 🖥️ 2. Workflow

The general workflow in ESPnet revolves around recipes.
A recipe is a structured and reproduceable set of steps for building, training, and evaluating a model for a specific task.
This includes things like data preparation, choice model architecture, training setup, and evaluation metrics.

This section will provide a high-level overview of how recipes work in ESPnet3 and how to create your own.

## 🗒️ 2.1. Recipes

Full Docs: [Recipes](./recipe_directory.md)

ESPnet3 recipes are located in the `egs3/` directory, which is structured as follows:

`egs3/<recipe>/<task>/`

For example, the ASR recipe for LibriSpeech 100h is located in `egs3/librispeech_100/asr`.

The task directory typically contains the following files and directories:
```
egs3/<recipe>/<task>/
├── run.py    # used to run the recipe
├── conf/     # YAML config files
├── dataset/  # code for downloading and building models
├── src/      # additional code
├── data/     # prepared manifests or recipe-local artifacts
└── exp/      # model checkpoints and experiment outputs
```

## 🧠 2.2. Systems

Full Docs: [Systems](./core/systems.md)

Training in ESPnet3 is driven by **System** classes.
Systems define the training structure for a specific task.
Currently, the following systems are supported:

| System | Task                         | Directory           |
| ------ | ---------------------------- | ------------------- |
| ASR    | automatic speech recognition | `egs3/TEMPLATE/asr` |
| TTS    | text-to-speech               | `egs3/TEMPLATE/tts` |


## 🪜 2.3. Stages

Each system divides training into several **stages** that are executed in order.
For example, a typical ASR pipeline consists of the following stages:

| Stage                                                | What it does                                                               |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| [`create_dataset`](./stages/create-dataset.md)       | download, validate, or materialize recipe-local dataset assets             |
| [`collect_stats`](./stages/collect-stats.md)         | compute shape files and dataset-level normalization stats                  |
| [`train`](./stages/train.md)                         | run training using PyTorch Lightning and write checkpoints under `exp_dir` |
| [`infer`](./stages/inference.md)                     | write SCP outputs under `inference_dir/<test_name>/`                       |
| [`measure`](./stages/measure.md)                     | read those outputs and write `metrics.json`                                |
| [`pack_model` / `upload_model`](./stages/publish.md) | package and optionally upload the trained model                            |

The default stage order for a given task is defined in:

```text
egs3/TEMPLATE/<task>/run.py
```

By default, all stages will be executed when running a recipe.
You can also run only the stages you need using the `--stages` flag:

```bash
python run.py \
  --stages train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

## ⚙️ 2.4. Configs

Full Docs: [Configs](./config/index.md)

Recipes are configured using the YAML files in the `conf/` directory.
Stages have separate config files and associated CLI flags:

| Stage area              | Config file                                      | CLI flag               |
| ----------------------- | ------------------------------------------------ | ---------------------- |
| training-related stages | [`training.yaml`](./config/train_config.md)      | `--training_config`    |
| inference               | [`inference.yaml`](./config/infer_config.md)     | `--inference_config`   |
| measurement             | [`metrics.yaml`](./config/measure_config.md)     | `--metrics_config`     |
| publication             | [`publication.yaml`](./config/publish_config.md) | `--publication_config` |

For a full list of configuration options, see the page for the relevant file. 

> [!IMPORTANT]
> Stages do not take arbitrary stage-specific CLI arguments. Keep stage settings
> inside YAML and pass the matching config file through `run.py`.
> You usually do not need to modify the system class just to pass a parameter into
> an existing stage.

## 🧪 2.5. Experiments

Running a recipe will create a sub-directory in `exp/` where the run's checkpoints, logs, and training outputs will be saved.
The name of the sub-directory (`exp_dir`) can be determined by setting the `exp_tag` key in `training.yaml` or `inference.yaml`, depending on which stages are run.
If `exp_tag` is not specified, it will be inherited from the TEMPLATE config.

In a combined training + inference run, inference inherits the training-side
experiment identity. In a standalone inference run, `inference.yaml` must define
its own identity through `exp_tag` or `exp_dir`.

### Example 1: training-driven naming

If `training.yaml` contains:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

then training outputs are written under:

```text
exp/training_branchformer/
```

If the same `run.py` invocation also runs inference, the runner copies
`exp_tag` and `exp_dir` into `inference_config`, so:

```yaml
inference_dir: ${exp_dir}/${self_name:}
```

becomes something like:

```text
exp/training_branchformer/inference
```

### Example 2: standalone inference naming

If inference runs by itself, `inference.yaml` must carry its own identity:

```yaml
exp_tag: whisper_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

That produces a directory such as:

```text
exp/whisper_eval/inference
```

### Example 3: naming resolution

One easy convention is:

```text
conf/
  tuning/
    training_e_branchformer.yaml
  decode/
    inference_beam5.yaml
```

with:

```yaml
# conf/tuning/training_e_branchformer.yaml
exp_tag: training_e_branchformer
```

and:

```yaml
# conf/decode/inference_beam5.yaml
exp_tag: inference_beam5
```

Then the corresponding outputs are easy to predict:

```text
exp/training_e_branchformer/
exp/inference_beam5/inference_beam5/
```

If inference is launched together with training, the training-side experiment
name takes priority and the decoding outputs stay under the training experiment
directory instead.

## 🧩 2.6. Putting Everything Together

To create a new recipe, start from:

```text
egs3/TEMPLATE/asr/run.py
```

Then define your recipe layout, write the configs, and run the stages you need.

Example:

```bash
cd egs3/<your_recipe>/<task>

# Dataset preparation
python run.py --stages create_dataset --training_config conf/training.yaml

# Optional stats + training
python run.py --stages collect_stats train --training_config conf/training.yaml

# Evaluation
python run.py \
  --stages infer measure \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Typical outputs go to:

- `exp/` for checkpoints, logs, and training outputs
- `inference_dir/` for inference outputs and `metrics.json`
- a packed publication directory for `pack_model`

### 💡 Tips for Working With Recipes
- Keep configs modular: dataset, model, trainer, and parallel blocks should stay separated.
- Choose experiment names deliberately so `exp_dir` stays readable.
- Use import-based execution when you want to integrate recipes into larger Python workflows.
- Reuse ESPnet2 task-backed model configs when possible, and add custom code only where the recipe actually differs.
- Prefer adding small helper code in `dataset/` or `src/` before inventing new stage interfaces.

# 📚 3. Additional ESPnet3 Documentation

### ✅ Cheat sheet: what you touch vs. what ESPnet3 provides

| Goal                               | You mainly edit or run                                    | Read next                                                                                                                                                         |
| ---------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Define datasets and builders       | `dataset/`, `training.yaml`, `create_dataset`             | [Datasets](./core/components/datasets.md), [Create dataset stage](./stages/create-dataset.md)                                                                     |
| Configure training                 | `training.yaml` for model, trainer, optimizer, dataloader | [Training config](./config/train_config.md), [Optimizer configuration](./core/components/optimizer_configuration.md), [Callbacks](./core/components/callbacks.md) |
| Run multi-GPU or cluster workloads | `training.yaml` and `parallel` blocks                     | [Multi-GPU / multi-node](./core/parallel/multiple_gpu.md), [Parallel](./core/parallel.md)                                                                         |
| Set up inference and measurement   | `inference.yaml` and `metrics.yaml`                       | [Inference](./stages/inference.md), [Measure](./stages/measure.md), [Provider / Runner](./core/parallel/provider_runner.md)                                       |
| Package a trained model            | `publication.yaml`                                        | [Publish stage](./stages/publish.md), [Publication config](./config/publish_config.md)                                                                            |

### Recipe Structure

- [Recipe directory layout](./recipe_directory.md)
- [Systems](./core/systems.md)


### Data & Datasets

- [Create dataset stage](./stages/create-dataset.md)
- [Dataset references and builders](./core/components/datasets.md)
- [DataOrganizer and dataset pipeline](./core/components/data-organizer.md)

### Training

- [Training stage](./stages/train.md)
- [Training config](./config/train_config.md)
- [Callbacks](./core/components/callbacks.md)
- [Optimizer configuration](./core/components/optimizer_configuration.md)
- [Multiple optimizers and schedulers](./core/components/multiple_optimizers_schedulers.md)
- [Multi-GPU / multi-node](./core/parallel/multiple_gpu.md)

### Inference & Evaluation

- [Inference stage](./stages/inference.md)
- [Inference config](./config/infer_config.md)
- [Measure stage](./stages/measure.md)
- [Measure config](./config/measure_config.md)

### Publication & Demos

- [Publication stage](./stages/publish.md)
- [Publication config](./config/publish_config.md)
- [Demo stage](./stages/demo.md)
- [Demo config](./config/demo_config.md)

### Execution Framework
- [Provider / Runner](./core/parallel/provider_runner.md)
- [Parallel configuration](./core/parallel.md)

### Developer Reference
- [Naming Conventions](./core/naming_convention.md)
- [System-specific stages](./stages/system-specific.md)
- [Components](./core/components.md)

# 📎 4. Reference Snippets

## Minimal recipe layout

```text
egs3/my_recipe/asr/
  run.py
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
  dataset/
    __init__.py
    builder.py
    dataset.py
  src/
    inference.py
  data/
  exp/
```

The most important convention is that config names, stage names, and directory
names line up:

- `training.yaml` drives `create_dataset`, `collect_stats`, and `train`
- `inference.yaml` drives `infer`
- `metrics.yaml` drives `measure`
- `publication.yaml` drives `pack_model` and `upload_model`

## `run.py`

```python
from espnet3.systems.asr.system import ASRSystem
from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, build_parser, main
from espnet3.utils.stages_utils import parse_cli_and_stage_args

parser = build_parser(stages=DEFAULT_STAGES)
args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

main(
    args=args,
    system_cls=ASRSystem,
    stages=DEFAULT_STAGES,
)
```

## `dataset/dataset.py`

```python
from pathlib import Path


class Dataset:
    def __init__(self, split: str, data_path: str | Path):
        self.split = split
        self.data_path = Path(data_path)

    def __getitem__(self, idx):
        return {
            "speech": self.data_path / self.split / "wav" / f"{idx}.wav",
            "text": "dummy text",
        }

    def __len__(self):
        return 100
```

## `dataset/builder.py`

```python
from pathlib import Path


class DatasetBuilder:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)

    def is_source_prepared(self) -> bool:
        return (self.dataset_dir / "downloads").exists()

    def prepare_source(self) -> None:
        ...

    def is_built(self) -> bool:
        return (self.dataset_dir / "train.jsonl").exists()

    def build(self) -> None:
        ...
```

The full dataset resolution and builder lifecycle are documented here:

- [Datasets](./core/components/datasets.md)
- [Create dataset stage](./stages/create-dataset.md)

## `src/inference.py`

```python
def build_output(*, data, model_output, idx):
    return {
        "utt_id": data.get("uttid", str(idx)),
        "hyp": model_output["text"],
    }
```

When batched inference is used, `data` becomes a list of samples and `idx`
becomes a list of indices.

## How `run.py` loads configs

ESPnet3 does not resolve interpolations immediately when it first reads a YAML
file. The current flow is:

1. load the packaged default config from `egs3/TEMPLATE/<task>/conf/`
2. load the recipe or user config
3. merge them
4. resolve the merged result once

```python
from espnet3.utils.config_utils import (
    load_and_merge_config,
    load_config_with_defaults,
)

training_config = load_and_merge_config(
    args.training_config,
    config_name="training.yaml",
    default_package=__package__,
    resolve=False,
)
inference_config = load_and_merge_config(
    args.inference_config,
    config_name="inference.yaml",
    default_package=__package__,
    resolve=False,
)
metrics_config = load_and_merge_config(
    args.metrics_config,
    config_name="metrics.yaml",
    default_package=__package__,
    resolve=False,
)
publication_config = load_config_with_defaults(args.publication_config)
```

## Resolve timing

Assume TEMPLATE `training.yaml` contains:

```yaml
recipe_dir: .
exp_tag: training
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

and a user config contains:

```yaml
exp_tag: training_e_branchformer
```

After merge and resolve, the result is:

```yaml
exp_tag: training_e_branchformer
exp_dir: ./exp/training_e_branchformer
```

## Minimal config examples

### `conf/training.yaml`

```yaml
# recipe override
exp_tag: training_branchformer
dataset_dir: ${recipe_dir}/data/mini_an4

dataset:
  train:
    - data_src: mini_an4/asr
      data_src_args:
        split: train
        data_path: ${dataset_dir}
  valid:
    - data_src: mini_an4/asr
      data_src_args:
        split: valid
        data_path: ${dataset_dir}

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
```

### `conf/inference.yaml`

```yaml
# recipe override
exp_tag: inference_beam5
dataset_dir: ${recipe_dir}/data/mini_an4

dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test
        data_path: ${dataset_dir}
```

Here `data_src: mini_an4/asr` means "resolve the dataset implementation from
that recipe tag, then pass `data_src_args` into `Dataset(...)`". You can also
use a dotted module path, or omit `data_src` entirely to load
`${recipe_dir}/dataset/__init__.py`.

## Stage-by-stage commands, configs, and outputs

### `create_dataset`

Run:

```bash
python run.py \
  --stages create_dataset \
  --training_config conf/training.yaml
```

Minimal config example:

```yaml
dataset_dir: ${recipe_dir}/data/mini_an4

create_dataset:
  recipe_dir: ${recipe_dir}
  dataset_dir: ${dataset_dir}
```

Typical output tree:

```text
data/
  mini_an4/
    downloads/
    train.jsonl
    valid.jsonl
    test.jsonl
```

### `collect_stats`

Run:

```bash
python run.py \
  --stages collect_stats \
  --training_config conf/training.yaml
```

Minimal config example:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
```

Typical output tree:

```text
exp/
  stats/
    train/
      feats_shape
      feats_stats.npz
    valid/
      feats_shape
      feats_stats.npz
```

### `train`

Run:

```bash
python run.py \
  --stages train \
  --training_config conf/training.yaml
```

Minimal config example:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

trainer:
  accelerator: auto
  devices: 1
```

Typical output tree:

```text
exp/
  training_branchformer/
    train.log
    last.ckpt
    epoch3_step12000_valid.loss.ckpt
    valid.loss.ave_3best.pth
    tensorboard/
```

### `infer`

Run:

```bash
python run.py \
  --stages infer \
  --inference_config conf/inference.yaml
```

If training and inference run together, pass both configs so inference inherits
the training-side experiment context:

```bash
python run.py \
  --stages train infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```

Minimal config example:

```yaml
exp_tag: inference_beam5
dataset_dir: ${recipe_dir}/data/mini_an4

dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test
        data_path: ${dataset_dir}
```

Typical output tree:

```text
exp/
  inference_beam5/
    inference/
      test/
        hyp.scp
        ref.scp
        utt_id.scp
```

Here `inference/` comes from the inference config name, while `test/` comes
from `dataset.test[].name`.

When inference runs inside the same `run.py` call as training, the tree is
usually:

```text
exp/
  training_branchformer/
    inference/
      test/
        hyp.scp
        ref.scp
        utt_id.scp
```

### `measure`

Run:

```bash
python run.py \
  --stages measure \
  --metrics_config conf/metrics.yaml
```

Minimal config example:

```yaml
exp_tag: training_branchformer
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/inference

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
```

Typical output tree after measurement:

```text
exp/
  training_branchformer/
    inference/
      test/
        hyp.scp
        ref.scp
        utt_id.scp
      metrics.json
```

### `pack_model`

Run:

```bash
python run.py \
  --stages pack_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

Minimal config example:

```yaml
pack_model:
  strategy: auto
  out_dir: ${recipe_dir}/exp/model_pack
  include_data_dir: true
```

Typical packed bundle tree:

```text
exp/
  model_pack/
    README.md
    meta.yaml
    scores.json
    conf/
      training.yaml
      inference.yaml
      metrics.yaml
      publication.yaml
    src/
      inference.py
    run.py
    exp/
      last.ckpt
      epoch3_step12000_valid.loss.ckpt
      valid.loss.ave_3best.pth
    data/
```

### `upload_model`

Run:

```bash
python run.py \
  --stages upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

Minimal config example:

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

## Useful next pages

- [Recipe directory layout](./recipe_directory.md)
- [Training config](./config/train_config.md)
- [Inference stage](./stages/inference.md)
- [Measure stage](./stages/measure.md)
- [Dataset references and builders](./core/components/datasets.md)
