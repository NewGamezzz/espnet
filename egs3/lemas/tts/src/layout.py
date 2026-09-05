"""Layout arithmetic shared by the training dataset and inference (spec 5).

Prompts are cut at 16 kHz in multiples of ``PROMPT_QUANTUM_16K`` samples, so
after 16 to 24 kHz resampling every prompt is a multiple of the 256-sample
hop and region boundaries fall exactly on mel frames. The whole concatenated
waveform gets ``n // HOP + 1`` frames (vocos, centre-padded); a region of
``n`` samples covers ``n // HOP`` frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.text.lemas_phonemizer import LANG_TOKEN, SPK_TOKEN, lang_tag

HOP = 256
SR = 24000
SRC_SR = 16000
PROMPT_QUANTUM_16K = 512  # 512 samples at 16 kHz = 768 at 24 kHz = 3 hops


def n_frames_total(n_samples: int) -> int:
    """Frames the vocos mel front-end yields for ``n_samples`` at 24 kHz."""
    return n_samples // HOP + 1


def region_frames(n_samples: int) -> int:
    """Frames covered by a hop-aligned region of ``n_samples``."""
    assert n_samples % HOP == 0, n_samples
    return n_samples // HOP


def quantize_prompt_16k(n_samples_16k: int) -> int:
    """Round a 16 kHz prompt length down to the prompt quantum."""
    return n_samples_16k - n_samples_16k % PROMPT_QUANTUM_16K


def cond_frames(spk_frames: int, lang_frames: int) -> int:
    """First target frame: the two prompt regions precede the target."""
    return spk_frames + lang_frames


class TokenTable:
    """Token to id mapping read from an espnet token list file."""

    def __init__(self, token_list_path):
        """Load the table.

        Args:
            token_list_path: One token per line; must contain ``<unk>``,
                ``<spk>``, ``<lang>`` and the language tags.

        Raises:
            KeyError: If a required special token is missing.

        Example:
            >>> table = TokenTable("data/tokens/tokens.txt")
            >>> table.id("<space>")
            4
        """
        with Path(token_list_path).open(encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.rstrip("\n")]
        self._id = {t: i for i, t in enumerate(tokens)}
        self.unk = self._id["<unk>"]
        self.spk = self._id[SPK_TOKEN]
        self.lang = self._id[LANG_TOKEN]
        self.size = len(tokens)

    def id(self, tok: str) -> int:
        """Id of ``tok``; unknown tokens map to ``<unk>``."""
        return self._id.get(tok, self.unk)

    def tag(self, lang: str) -> int:
        """Id of the language tag ``<lang>``, e.g. ``<de>``."""
        return self._id[lang_tag(lang)]


def build_text_ids(
    spk_frames: int,
    lang_frames: int,
    lang: str,
    phones: Sequence[str],
    table: TokenTable,
) -> np.ndarray:
    """Build the text lane: role tokens per prompt frame, then tag + phones.

    Args:
        spk_frames: Frames of the speaker prompt region (0 when absent).
        lang_frames: Frames of the language prompt region (0 when absent).
        lang: Target language code (selects the tag token).
        phones: Target phone tokens.
        table: Token table.

    Returns:
        ``int64`` id array of length ``spk_frames + lang_frames + 1 + len(phones)``.

    Example:
        >>> build_text_ids(2, 1, "de", ["a"], table).tolist()
        [5, 5, 6, 7, 2]
    """
    ids = [table.spk] * spk_frames + [table.lang] * lang_frames + [table.tag(lang)]
    ids.extend(table.id(p) for p in phones)
    return np.asarray(ids, dtype=np.int64)
