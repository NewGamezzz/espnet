"""Test setup: put the worktree root and the recipe dir on sys.path."""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parents[1]
WORKTREE = RECIPE_DIR.parents[2]
for p in (str(WORKTREE), str(RECIPE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dataset.manifest import ManifestRow, write_manifest  # noqa: E402

TOKENS = [
    "<blank>",
    "<unk>",
    "a",
    "b",
    "<space>",
    "<spk>",
    "<lang>",
    "<de>",
    "<zh>",
    "<sos/eos>",
]


@pytest.fixture
def corpus(tmp_path):
    """A tiny packed corpus: two .pcm packs (de, zh) and a manifest over them."""
    audio = tmp_path / "audio"
    packs: dict = {}
    rows = []

    def clip(pack, sec):
        n = int(sec * 16000)
        wav = 0.1 * np.sin(np.arange(n) * 0.05)
        buf = packs.setdefault(pack, bytearray())
        start = len(buf) // 2
        buf += (wav * 32768).astype("<i2").tobytes()
        return f"{pack}:{start}:{n}"

    # de: one video group with 4 segments, one singleton video,
    # one mls speaker with 2 rows
    for i in range(4):
        rows.append(
            ManifestRow(
                f"de_vidAAAAAAAA-0000{i}-00000000-00000300",
                clip("de/d.pcm", 3.0),
                "a b <space> a",
                "de",
                "yodas",
                "vidAAAAAAAA",
                3.0,
                "j",
                0,
                "group",
                "",
                "",
            )
        )
    rows.append(
        ManifestRow(
            "de_vidBBBBBBBB-00000-00000000-00000200",
            clip("de/d.pcm", 2.0),
            "b",
            "de",
            "yodas",
            "vidBBBBBBBB",
            2.0,
            "j",
            0,
            "none",
            "",
            "",
        )
    )
    for i in range(2):
        rows.append(
            ManifestRow(
                f"de_77_1_00000{i}",
                clip("de/d.pcm", 8.0),
                "a a",
                "de",
                "mls",
                "77",
                8.0,
                "j",
                0,
                "group",
                "",
                "",
            )
        )
    # zh: split rows (no group)
    for i in range(2):
        rows.append(
            ManifestRow(
                f"zh_emilia_zh_000000000{i}",
                clip("zh/z.pcm", 6.0),
                "a b a b a b",
                "zh",
                "emilia",
                "",
                6.0,
                "j",
                0,
                "split",
                "0.1:1.0,1.1:2.0,2.1:3.0,3.1:4.0,4.1:5.0,5.1:5.9",
                "a|b|a|b|a|b",
            )
        )
    for pack, buf in packs.items():
        path = audio / pack
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(buf))
    m = tmp_path / "train.tsv"
    write_manifest(rows, m)
    tok = tmp_path / "tokens.txt"
    tok.write_text("\n".join(TOKENS) + "\n")
    return dict(manifest=m, tokens=tok, audio=audio)
