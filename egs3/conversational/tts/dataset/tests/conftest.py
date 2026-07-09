"""Shared path setup and fixtures for the dataset test suite.

Tests import the code under test through its package path
(``egs3.conversational.tts.dataset``) so config loading via
``importlib.resources.files(__package__)`` behaves as in real use.
"""

import string
import sys
from collections import namedtuple
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[5]  # espnet repo root, so egs3/espnet2/espnet3 resolve locally
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Duck-typed stand-in for sssd.Turn: text.py/windows.py only touch attributes.
FakeTurn = namedtuple("FakeTurn", ["channel", "speaker", "text", "start", "end"])


@pytest.fixture
def turns_3spk() -> list[FakeTurn]:
    """Fabricated 3-speaker turn list in conversation order."""
    return [
        FakeTurn(0, "spk_a", "good afternoon. how are you?", 0.5, 3.1),
        FakeTurn(1, "spk_b", "good. what about you?", 3.6, 5.2),
        FakeTurn(2, "spk_c", "hi there", 5.9, 6.7),
        FakeTurn(0, "spk_a", "good, but i have a problem", 7.4, 9.8),
        FakeTurn(1, "spk_b", "oh no", 10.5, 11.0),
    ]


@pytest.fixture
def base_vocab() -> list[str]:
    """Tiny char vocab in char_tokens.txt format (line index = token id)."""
    return ["<blank>", "<unk>", "<space>"] + list(string.ascii_lowercase) + [
        ".",
        ",",
        "?",
        "!",
        "'",
        "<sos/eos>",
    ]
