"""Shared path setup and fixtures for the dataset test suite.

Tests import the code under test through its package path
(``egs3.conversational.tts.dataset``) so config loading via
``importlib.resources.files(__package__)`` behaves as in real use.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[5]  # espnet repo root, so egs3/espnet2/espnet3 resolve locally
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
