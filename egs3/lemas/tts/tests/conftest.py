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
import soundfile as sf  # noqa: E402
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
    audio = tmp_path / "audio"
    rows = []

    def clip(rel, sec):
        p = audio / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        wav = 0.1 * np.sin(np.arange(int(sec * 16000)) * 0.05).astype(np.float32)
        sf.write(p, wav, 16000, format="FLAC", subtype="PCM_16")

    # de: one video group with 4 segments, one singleton video,
    # one mls speaker with 2 rows
    for i in range(4):
        clip(f"de/d/v1_{i}.flac", 3.0)
        rows.append(
            ManifestRow(
                f"de_vidAAAAAAAA-0000{i}-00000000-00000300",
                f"de/d/v1_{i}.flac",
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
    clip("de/d/v2_0.flac", 2.0)
    rows.append(
        ManifestRow(
            "de_vidBBBBBBBB-00000-00000000-00000200",
            "de/d/v2_0.flac",
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
        clip(f"de/d/m_{i}.flac", 8.0)
        rows.append(
            ManifestRow(
                f"de_77_1_00000{i}",
                f"de/d/m_{i}.flac",
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
        clip(f"zh/z/s_{i}.flac", 6.0)
        rows.append(
            ManifestRow(
                f"zh_emilia_zh_000000000{i}",
                f"zh/z/s_{i}.flac",
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
    m = tmp_path / "train.tsv"
    write_manifest(rows, m)
    tok = tmp_path / "tokens.txt"
    tok.write_text("\n".join(TOKENS) + "\n")
    return dict(manifest=m, tokens=tok, audio=audio)
