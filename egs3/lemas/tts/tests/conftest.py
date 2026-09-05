"""Test setup: put the worktree root and the recipe dir on sys.path."""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parents[1]
WORKTREE = RECIPE_DIR.parents[2]
for p in (str(WORKTREE), str(RECIPE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
