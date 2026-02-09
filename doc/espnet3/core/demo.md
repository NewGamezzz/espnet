---
title: ESPnet3 Demo
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🎛 ESPnet3 Demo (`espnet3/demo`) — developer landing

ESPnet3 demos are *packaged inference apps* generated from configuration. The
demo runtime reuses your existing inference model/provider/runner logic, so you
usually do not need demo-specific Python.

This page explains the architecture and extension points. For end-to-end usage
and full config details, see:

- [Demo guide](../stages/demo.md)
- [Demo configuration](../config/demo_config.md)

## ✅ What a demo is in ESPnet3

A demo consists of:

- a `demo.yaml` that defines UI + runtime wiring
- an `infer.yaml` used to build the model and (optionally) control output mapping
- a generated app entrypoint (`app.py`) plus packaged assets

Packing is done by stages:

- `pack_demo`: generate a runnable demo directory
- `upload_demo`: upload the packed demo (e.g., to a remote hosting target)


## 🧩 Key modules (where to look)

| Module | Description | Detailed doc |
| --- | --- | --- |
| `espnet3/demo/pack.py` | Pack demo directory (`pack_demo`, `upload_demo`). | [Pack pipeline](./components/demo/pack_pipeline.md) |
| `espnet3/demo/setup.py` | Generate `app.py` and `requirements.txt` during packing. | [Pack pipeline](./components/demo/pack_pipeline.md) |
| `espnet3/demo/app_builder.py` | Build the Gradio app from `demo.yaml` and assets. | [App builder](./components/demo/app_builder.md) |
| `espnet3/demo/ui.py` | Convert the `ui:` section in `demo.yaml` into Gradio components. | [UI definition](./components/demo/ui_definition.md) |
| `espnet3/demo/runtime.py` | Load model, build SingleItemDataset, run inference, map outputs. | [Runtime](./components/demo/runtime.md) |
| `espnet3/demo/resolve.py` | Resolve provider/runner classes and system defaults. |  |

## ⚙️ Resolution and runtime wiring

### Provider / runner selection

The demo runtime resolves the inference classes in this order:

1. `inference.provider_class` / `inference.runner_class` in `demo.yaml` (if set)
2. Convention by `system` in `demo.yaml`:
   - `espnet3.systems.<system>.inference.InferenceProvider`
   - `espnet3.systems.<system>.inference.InferenceRunner`
3. Fallback to the generic implementations:
   - `espnet3.systems.base.inference_provider.InferenceProvider`
   - `espnet3.systems.base.inference_runner.InferenceRunner`

Implementation: `espnet3/demo/resolve.py` (`resolve_provider_class`, `resolve_runner_class`).

### Notes

Provider/runner *class selection* is driven by `demo.yaml` (and system
conventions), not by `infer.yaml`. `infer.yaml` is used at runtime to:

- build the model (`provider_cls.build_model(infer_cfg)`)
- optionally forward keys like `infer_config.input_key` and `infer_config.output_fn_path`
  into runner kwargs (see `espnet3/demo/runtime.py:run_inference`)

### UI resolution

UI config comes from:

1. the `ui:` section in `demo.yaml` merged onto system defaults (if available)
2. otherwise, system `build_ui(demo_cfg)` / `build_ui_default()` hooks

Implementation: `espnet3/demo/app_builder.py`.

### How the runner is called

The demo runtime wraps the UI inputs into a **single-item dataset** and then
invokes one inference pass:

- dataset: `SingleItemDataset({<ui_name>: <ui_value>, ...})`
- call: `runner_cls.forward(0, dataset=dataset, model=model, **extra_kwargs)`

Implementation: `espnet3/demo/runtime.py`.

## 🗝 Output mapping (`output_keys`)

The demo UI expects a list of output values. ESPnet3 maps the runner/model
result to UI outputs using `demo.output_keys` (or system defaults):

```yaml
output_keys:
  text: hyp
```

This means: fill the UI output component named `text` with `result["hyp"]`.

If UI outputs exist but `output_keys` is missing/mismatched, the runtime raises
an error (see `espnet3/demo/runtime.py:_map_outputs`).

## 🛠 Customization checklist

- Change UI layout: edit the `ui:` section in `demo.yaml` (or implement system defaults)
- Change inference logic: set `inference.provider_class` / `inference.runner_class` in `demo.yaml`
- Pass constant decoding args: set `extra_kwargs:` in `demo.yaml`
- Package additional files: use `pack.files:` / `pack.out_dir:` in `demo.yaml`

## 🧱 Adding demo support for a new system

To make a system "demo-friendly" without requiring every recipe to specify
everything explicitly:

- implement `espnet3.systems.<system>.demo` with:
  - `build_ui_default()` and/or `build_ui(demo_cfg)`
  - `build_inference_default()` (for default `output_keys` / `extra_kwargs`)
- follow (or explicitly override) the provider/runner import conventions

If your system does not follow conventions, require recipes to set
`inference.provider_class` / `inference.runner_class` in `demo.yaml`.
