"""Fail a job early when python cannot import the recipe from this checkout.

The shared environment on Delta has a different espnet checkout installed.
If PYTHONPATH is wrong, ``espnet3`` and ``egs3`` resolve from that checkout
while ``src/`` and ``dataset/`` load from this one, and the job dies hours
later (or, worse, trains with the other tree's dataloader). A module that
cannot be imported at all raises here, with its traceback.

Usage: python local/delta/check_imports.py <repo root>
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# espnet2.tasks.tts is the training task: importing it here also fails a job
# at once when the env lacks a package only the model code needs (pyworld).
MODULES = ("espnet2", "espnet3", "egs3.emilia.tts", "espnet2.tasks.tts")


def foreign_modules(repo_root: str | Path) -> list[tuple[str, str]]:
    """Return ``(module, file)`` for each module not imported from the repo."""
    root = Path(repo_root).resolve()
    foreign = []
    for name in MODULES:
        origin = Path(importlib.import_module(name).__file__).resolve()
        if root not in origin.parents:
            foreign.append((name, str(origin)))
    return foreign


if __name__ == "__main__":
    wrong = foreign_modules(sys.argv[1])
    for module, origin in wrong:
        print(f"{module} is imported from {origin}, not {sys.argv[1]}", file=sys.stderr)
    sys.exit(1 if wrong else 0)
