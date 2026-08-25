"""Shared helpers for asserting that every dotted path a config names
actually imports.

Motivation (IMPORTANT 5 in the final whole-branch review): C1 (`infer`)
and C2 (`measure`) both shipped configs pointing at Python objects that did
not exist (`src.inference.build_output`, `src.metrics.versa.VersaMetric`),
and every task-scoped review passed because no test ever imported what the
configs named -- `test_seedtts_inference_config.py` checked `_target_`,
`data_src`, `device` and `train_config` as plain strings, never resolving
`output_fn`, and `metrics.yaml` had no test at all.

These helpers resolve a dotted path exactly the way the production code
does, so a test built on top of them is a real gate, not an approximation:

- `resolve_dotted_path` mirrors
  `espnet3.systems.base.inference_runner._load_output_fn`: split on the
  last dot, `import_module` the prefix, `getattr` the suffix. This is also
  how `espnet3/systems/base/inference.py:172-174` resolves `output_fn`
  before the `infer` stage ever runs.
- `iter_targets` walks a (nested) config/container and yields every
  `_target_` string found, so a test can resolve all of them without
  hand-listing each one.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterator


def resolve_dotted_path(path: str) -> Any:
    """Import ``path`` the same way `_load_output_fn` and hydra's
    `_target_` resolution do: split on the last dot, import the module,
    getattr the attribute.

    Raises ImportError / AttributeError directly -- the whole point is to
    fail loudly when a config names something that does not exist.
    """
    module_path, name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, name)


def iter_targets(container: Any) -> Iterator[str]:
    """Yield every `_target_` string found anywhere inside `container`.

    `container` may be a (possibly nested) dict/list, as returned by
    `OmegaConf.to_container(cfg, resolve=True)`.
    """
    if isinstance(container, dict):
        target = container.get("_target_")
        if isinstance(target, str):
            yield target
        for value in container.values():
            yield from iter_targets(value)
    elif isinstance(container, list):
        for item in container:
            yield from iter_targets(item)
