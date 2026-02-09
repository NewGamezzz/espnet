---
title: Demo UI Definition
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🧩 Demo UI component catalog and extension

This page is for developers who want to **define/extend the supported demo UI
component types** (the building blocks used by `demo.yaml ui:`). If you only
want to edit demo layouts, `doc/vuepress/src/espnet3/stages/demo.md` and
`doc/vuepress/src/espnet3/config/demo_config.md` are usually enough.

## 📍 Where supported UI parts are defined

All supported UI component `type` values are defined in one place:

- `espnet3/demo/ui.py:_build_component`

The config-to-UI pipeline is:

- `espnet3/demo/app_builder.py:_load_ui_spec` (loads `demo.yaml` + merges defaults)
- `espnet3/demo/ui.py:build_ui_from_config` (parses `ui:` and builds components)
- `espnet3/demo/ui.py:_build_components` / `_build_component` (instantiates Gradio)

If you need to add a new UI part, you will edit `espnet3/demo/ui.py`.

## 🧱 Base schema

Every entry under `ui.inputs` / `ui.outputs` in `demo.yaml` must include:

- `name` (non-empty string; used as the runtime key)
- `type` (one of the supported values below)

Optional (all types):

- `label` (falls back to `name`)

## 🧩 Supported component types

Implemented in `espnet3/demo/ui.py:_build_component`:

| type | Gradio component | Supported keys (besides `name`/`type`) | Notes |
| --- | --- | --- | --- |
| `audio` | `gradio.Audio` | `label`, `sources`, `audio_type` | `sources` supports `mic\|microphone`, `upload\|file` |
| `textbox` | `gradio.Textbox` | `label`, `lines`, `placeholder` | `lines` default: `1` |
| `dropdown` | `gradio.Dropdown` | `label`, `choices`, `value` |  |
| `number` | `gradio.Number` | `label`, `value` |  |
| `slider` | `gradio.Slider` | `label`, `min`, `max`, `step`, `value` |  |
| `checkbox` | `gradio.Checkbox` | `label`, `value` |  |
| `image` | `gradio.Image` | `label` |  |
| `file` | `gradio.File` | `label` |  |

Unknown types raise `ValueError("Unsupported UI component type: ...")`.

## 🧪 Adding a new component type

Use this as a checklist:

- Add a new `type` branch in `espnet3/demo/ui.py:_build_component` that returns a Gradio component.
- Verify runtime wiring still matches expectations:
  - UI `name` becomes a key in `SingleItemDataset`
  - outputs map via `output_keys` (`espnet3/demo/runtime.py:_map_outputs`)
- Add a small config example to `doc/vuepress/src/espnet3/stages/demo.md` if users should use it.

## ✅ Design constraints

- Keep component creation pure and config-driven (no external state).
- Prefer backward-compatible changes (new keys optional, sensible defaults).
- Avoid changing return types for existing components (breaks demos silently).
