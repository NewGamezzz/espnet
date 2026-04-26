---
title: Getting Started with ESPnet3
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-04-24
---

# ESPnet3 Experiment Naming Examples

Below are some of examples of how experiment names are determined based on the YAML configs.

---

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

---

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

---

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