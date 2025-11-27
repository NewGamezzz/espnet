---
title: Getting Started with ESPnet3
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🚀 Getting Started with ESPnet3

This guide provides the fastest way to start using ESPnet3.  
Choose the workflow that fits your environment and follow the examples below.

---

# ⚡ Quick Start (ASR Example)

## 1. Install ESPnet3

ESPnet3 is distributed under the **same pip package name: `espnet`**.

```bash
pip install espnet
````

### Install from source (recommended for development)

```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools

# Recommended: setup_uv.sh  
# Installs pixi + uv and sets up all dependencies much faster than conda.
. setup_uv.sh
```

---

## 📦 2. Install system-specific dependencies

ESPnet3 introduces the concept of a **system** (ASR, TTS, ST, ENH, etc.).
Each system may require additional packages not used by others.

Install system extras using:

```bash
pip install "espnet[asr]"
```

Other examples:

```bash
pip install "espnet[tts]"
pip install "espnet[st]"
pip install "espnet[enh]"
```

If installed from a cloned repository:

```bash
pip install -e ".[asr]"
# or using uv:
uv pip install -e ".[asr]"
```

---

## 🧪 3. Run a recipe **without cloning the repository**

(import-based execution)

ESPnet3 recipes are fully importable.
Create config files locally and run:

```python
from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, main
from espnet3.systems.asr.system import ASRSystem

main(
    train_config="/path/to/train_config.yaml",
    eval_config="/path/to/eval_config.yaml",
    system_cls=ASRSystem,
    stages=DEFAULT_STAGES,
    stage_configs={
        "stage_name": {
            "key": "value"
        }
    }
)
```

This is useful for programmatic pipelines or MLOps workflows.

---

## 🖥 4. Run a recipe **with a cloned repository**

All configs and scripts live inside `egs3/`.

Example: LibriSpeech 100h ASR

```bash
cd egs3/librispeech_100/asr
python run.py \
    --stage all \
    --train_config conf/train.yaml \
    --eval_config conf/evaluate.yaml
```

---

# 🧠 Understanding Stages

The default stage order is defined in:

```
egs3/TEMPLATE/<system>/run.py
```

Typical ASR pipeline:

1. **create_dataset** (download/prepare raw data)
2. **collect_stats** (compute CMVN/statistics)
3. **train** (fit the model)
4. **evaluate**

   * **decode** (generate hypotheses)
   * **score** (compute metrics)
5. **publish** (package artifacts)

You can run selected stages:

```bash
python run.py \
    --stage train evaluate \
    --train_config conf/train.yaml \
    --eval_config conf/evaluate.yaml
```

---

# 🧵 Passing arguments to specific stages

Each stage accepts arguments via namespaced CLI flags:

```
--stage.<stage_name>.<argument> <value>
```

Example (ASR):

```bash
python run.py \
  --stage all \
  --train_config conf/train.yaml \
  --eval_config conf/evaluate.yaml \
  --stage.create_dataset.dataset_root /path/to/LibriSpeech
```

Where `src/create_dataset.py` contains:

```python
def create_dataset(dataset_root, ...):
    ...
```

Your custom stages also automatically receive arguments:

```
--stage.custom_stage.some_arg value
```

No code changes inside the system class are needed.

---

# 🧩 Implementing `src/` for your recipe

Each recipe may define custom logic inside:

```
egs3/<recipe>/<task>/src/
```

Typical files:

* **create_dataset.py** - dataset preparation functions
* **dataset.py** - dataset builder or transform classes
* **custom_model.py** - user-defined modules referenced by Hydra configs

`run.py` automatically adds this directory to `PYTHONPATH`.

---

# ⚙️ Configurations (Hydra)

All hyperparameters live in `conf/*.yaml`.

**Important: ESPnet3 disables CLI overrides (`--key=value`).**
This is because ESPnet3 relies on hierarchical config merging that conflicts
with Hydra’s runtime override semantics.
All overrides must be written inside YAML files.

---

# ✅ Putting Everything Together (cloned repository workflow)

Start from:

```
egs3/TEMPLATE/asr/run.py
```

Replace `ASRSystem` if you define a custom system. Then:

```bash
cd egs3/<your_recipe>/<task>

# Dataset preparation
python run.py --stage create_dataset --train_config conf/train.yaml

# (Optional) collect_stats + training
python run.py --stage train --train_config conf/train.yaml --collect_stats

# Evaluation
python run.py --stage evaluate --eval_config conf/evaluate.yaml
```

Outputs go to:

* `exp/` – training logs + checkpoints
* `decode_*/` – decoding + scoring results

---

## 📚 Additional ESPnet3 Documentation

### ✅ Cheat sheet: what you touch vs. what’s provided

| Goal                        | You mainly edit / run                        | Read next                          |
| --------------------------- | -------------------------------------------- | ---------------------------------- |
| Define datasets and loaders | `conf/dataset*.yaml`, `DataOrganizer` config | `dataset.md`                       |
| Configure training          | `conf/train.yaml` (model, trainer, optim)    | `optimizer_configuration.md`, `callbacks.md` |
| Run multi-GPU / cluster     | `conf/train.yaml` + `parallel` blocks        | `multiple_gpu.md`, `config.md`     |
| Set up evaluation           | `conf/evaluate.yaml` and score scripts       | `evaluate.md`, `provider_runner.md`|

### Execution Framework

* **Provider / Runner:** `provider_runner.md`
* **Configuration:** `parallel.md`

### Data & Datasets

* **Data preparation examples:** `data_preparation.md`
* **Dataset classes & sharding:** `dataset.md`

### Training

* **Callbacks:** `callbacks.md`
* **Optimizers:** `optimizer_configuration.md`
* **Multiple optimizers/schedulers:** `multiple_optimizers_schedulers.md`
* **Multi-GPU & multi-node:** `multiple_gpu.md`

### Inference & Evaluation

* **Runner-based decoding:** `provider_runner.md`
* **Scoring pipeline:** `evaluate.md`

### Recipe Structure

* **Recipe directory layout:** `recipe_directory.md`
* **System entry points:** `system.md`

---

## 💡 Tips for Working With Recipes

* Keep configs modular: dataset / model / trainer / parallel blocks.
* Decide early whether execution is **local** or **SLURM/cluster**.
* Use import-based execution for MLOps pipelines.
* Reuse ESPnet2 model configs where possible.
