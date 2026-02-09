---
title: Demo Pack Pipeline
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 📦 Demo pack pipeline

Goal: explain how `pack_demo` produces a runnable `demo/` directory, including
where `app.py` comes from and what files are written/copied/symlinked.

## 🔗 Entry points

- Stage: `pack_demo` → `espnet3/systems/base/system.py:System.pack_demo`
- Implementation: `espnet3/demo/pack.py:pack_demo`

## 🗂 Output layout

`pack_demo` produces a runnable directory. Typical output:

```text
demo
├── app.py
├── demo.yaml
├── requirements.txt
├── README.md
├── config
│   └── infer.yaml
├── data -> ../data/
├── exp -> ../exp/
└── (optional files from pack.files ...)
```

Notes:

- `demo.yaml` is rewritten so `infer_config` points to `config/infer.yaml` and
  `demo_dir` is set.
- `README.md` may be generated if missing.
- Entries under `pack.files` in `demo.yaml` are typically added as symlinks
  into the demo directory (implementation detail in `espnet3/demo/pack.py`).

## 🏭 How each file is generated

- `app.py`
  - Written by `espnet3/demo/setup.py:_write_app_py()` via
    `espnet3/demo/setup.py:setup_demo_assets()`
  - Contents come from `espnet3/demo/setup.py:_APP_TEMPLATE` and call
    `espnet3/demo/app_builder.py:build_demo_app(demo_dir)`

- `requirements.txt`
  - Written by `espnet3/demo/setup.py:_write_requirements()` via
    `espnet3/demo/setup.py:setup_demo_assets()`
  - Uses `requirements:` in `demo.yaml` if present, otherwise writes a small default
  - To add system/recipe-specific dependencies (e.g., extra tokenizers, GPU libs),
    set `requirements:` in `demo.yaml` to the full list you want shipped with
    the demo package. `pack_demo` will write it into the packed directory.

- `demo.yaml`
  - Prepared by `espnet3/demo/pack.py:_prepare_demo_config()`:
    - sets `demo_dir`
    - rewrites `infer_config` to `config/infer.yaml` for the packed demo
    - sets `ui.article_path` to `README.md` when missing
  - Written by `espnet3/demo/pack.py:_write_demo_config()`

- `config/infer.yaml`
  - Loaded/resolved from the source `infer_config` by:
    - `espnet3/demo/resolve.py:resolve_infer_path()`
    - `espnet3/demo/resolve.py:load_infer_config()`
  - Written to the demo directory by `espnet3/demo/pack.py:_write_infer_config()`

- `README.md`
  - Created only if missing by `espnet3/demo/pack.py:_write_readme_if_missing()`
  - Template resolution happens in `espnet3/demo/pack.py:_load_demo_readme_template()`

- `data`, `exp`, and any other entries under `pack.files` in `demo.yaml`
  - Added by `espnet3/demo/pack.py:_copy_pack_files()` (typically as symlinks via
    `espnet3/demo/pack.py:_copy_path(..., symlink=True)`)
