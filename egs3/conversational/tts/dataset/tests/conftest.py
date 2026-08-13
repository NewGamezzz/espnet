"""Shared path setup and fixtures for the dataset test suite.

Tests import the code under test through its package path
(``egs3.conversational.tts.dataset``) so config loading via
``importlib.resources.files(__package__)`` behaves as in real use.
"""

import math
import string
import sys
from collections import namedtuple
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[
    5
]  # espnet repo root, so egs3/espnet2/espnet3 resolve locally
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


def channel_tone_hz(channel: int) -> float:
    """Per-channel pure-tone frequency, so channel identity is spectrally
    detectable after permutation and resampling."""
    return 1000.0 * (channel + 1)


def write_flac(
    path: Path, num_channels: int, duration_s: float, sr: int = 48000
) -> None:
    """Write a synthetic FLAC where channel i is a pure tone at channel_tone_hz(i)."""
    import numpy as np
    import soundfile as sf

    t = np.arange(int(round(duration_s * sr))) / sr
    data = np.stack(
        [
            0.3 * np.sin(2 * math.pi * channel_tone_hz(c) * t)
            for c in range(num_channels)
        ],
        axis=1,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="FLAC")


@pytest.fixture
def base_vocab() -> list[str]:
    """Tiny char vocab in char_tokens.txt format (line index = token id)."""
    return (
        ["<blank>", "<unk>", "<space>"]
        + list(string.ascii_lowercase)
        + [
            ".",
            ",",
            "?",
            "!",
            "'",
            "<sos/eos>",
        ]
    )


def _alternating_sups(session_id, num_channels, duration, utt_len=2.5, gap=1.5):
    """Round-robin utterances across channels with clean inter-turn silences."""
    sups = []
    t, i = 0.5, 0
    texts = ["Can you hear me?", "yes, loud and clear!", "great. let's start"]
    while t + utt_len + 0.5 < duration:
        channel = i % num_channels
        sups.append(
            {
                "id": f"{session_id}_ch{channel}_utt{i:04d}",
                "recording_id": session_id,
                "start": round(t, 3),
                "duration": utt_len,
                "channel": channel,
                "text": texts[i % len(texts)],
                "speaker": f"{session_id}_spk{channel}",
            }
        )
        t += utt_len + gap
        i += 1
    return sups


# Shared session table for the SSSD-shaped fixture tree: session id, channel
# count, duration.  Both the ``fake_corpus`` fixture (recordings/supervisions
# for the real builder pipeline) and tests that need SessionRecords directly
# (see test_dataset.py's ``_sessions_from_fixture``) derive from this single
# source so the two stay in sync.
FAKE_SESSIONS = [("sess_long", 2, 60.0), ("sess_tri", 3, 40.0), ("sess_short", 2, 8.0)]


@pytest.fixture
def fake_corpus(tmp_path, base_vocab):
    """Miniature SSSD corpus tree + base vocab file + empty recipe dir."""
    import gzip
    import json

    root = tmp_path / "corpus"
    sessions = FAKE_SESSIONS
    recordings, supervisions = [], []
    for session_id, num_channels, duration in sessions:
        write_flac(
            root / "original" / f"{session_id}_mixed.flac", num_channels, duration
        )
        recordings.append(
            {
                "id": session_id,
                "sources": [
                    {
                        "type": "file",
                        "channels": list(range(num_channels)),
                        # Absolute prefix is bogus on purpose: the loader must
                        # remap onto dataset_root instead of trusting it.
                        "source": (
                            f"/scratch/elsewhere/original/{session_id}_mixed.flac"
                        ),
                    }
                ],
                "sampling_rate": 48000,
                "num_samples": int(duration * 48000),
                "duration": duration,
                "channel_ids": list(range(num_channels)),
            }
        )
        supervisions.extend(_alternating_sups(session_id, num_channels, duration))

    manifests = root / "lhotse_manifests_48"
    manifests.mkdir(parents=True)
    with gzip.open(manifests / "recordings.jsonl.gz", "wt", encoding="utf-8") as f:
        for rec in recordings:
            f.write(json.dumps(rec) + "\n")
    with gzip.open(manifests / "supervisions.jsonl.gz", "wt", encoding="utf-8") as f:
        for sup in supervisions:
            f.write(json.dumps(sup) + "\n")

    vocab_path = tmp_path / "base_vocab.txt"
    vocab_path.write_text("\n".join(base_vocab) + "\n", encoding="utf-8")
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    return {"root": root, "base_vocab_path": vocab_path, "recipe_dir": recipe_dir}
