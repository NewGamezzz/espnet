---
title: Demo App Builder
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🧰 `app_builder.py` overview

This page explains how `demo.yaml` + packed assets turn into a runnable Gradio
app, and where to patch behavior as a developer.

`espnet3/demo/app_builder.py` is primarily a **library module used by the
generated `demo/app.py`** created during `pack_demo`. It exists to make the
packed demo directory runnable via a minimal, stable entrypoint.

## 🧷 How it is invoked from the packed demo

When you run the packed demo (`python app.py`), the generated `app.py` calls `build_demo_app` with the demo
directory path.

Generation and call path:

- `pack_demo` → `espnet3/demo/setup.py:_write_app_py` writes `demo/app.py`
- `demo/app.py` (generated) calls:

```python
from espnet3.demo.app_builder import build_demo_app

build_demo_app(Path(__file__).resolve().parent)
```

So `build_demo_app(demo_dir)` should treat `demo_dir` as the root that contains
`demo.yaml`, `config/infer.yaml`, and `README.md`.

High-level flow:

1. load `demo.yaml`
2. build runtime (`espnet3/demo/runtime.py:build_runtime`)
3. resolve UI spec (`_load_ui_spec`)
4. create Gradio Blocks and connect `button.click` to `run_inference`

## 🧩 Key behavior

### 🧬 UI resolution and merge

`_load_ui_spec(demo_cfg, demo_dir)` decides where UI comes from:

- If `demo.yaml` has no `ui:` section, it tries system hooks
  (`espnet3.systems.<system>.demo.build_ui`).
- If `demo.yaml` has `ui:`, it merges it onto system defaults
  (`build_ui_default`) when available.

### 🧱 Component instantiation

After UI config is chosen, components are created by:

- `espnet3/demo/ui.py:build_ui_from_config`
- `espnet3/demo/ui.py:_build_component`

This is where supported `type` values are defined.

### 📄 README rendering in the demo page

During packing, `pack_demo` sets `ui.article_path: README.md` when not provided
(`espnet3/demo/pack.py:_prepare_demo_config`). At runtime, `build_demo_app`
reads that file and renders it as Markdown **inside the demo UI**:

- `espnet3/demo/app_builder.py:_read_article`

This is why `demo/README.md` shows up in the demo screen.

To change what is rendered in the demo page, edit the packed `demo/README.md`
directly (or set `ui.article_path` to a different Markdown file in `demo.yaml`).

### 🔌 Inference callback wiring

`build_demo_app` defines a `_predict(*values)` callback and wires it to the Run
button:

- callback: `espnet3/demo/runtime.py:run_inference`
- inputs: UI component values + names
- outputs: UI output component names (for `output_keys` mapping)

## 🛠 What to modify 

- Add a new UI component type: edit `espnet3/demo/ui.py:_build_component`
- Change UI resolution/merge rules: edit `espnet3/demo/app_builder.py:_load_ui_spec`
- Change how README/article is injected: edit `espnet3/demo/app_builder.py` and/or
  `espnet3/demo/pack.py:_prepare_demo_config`
