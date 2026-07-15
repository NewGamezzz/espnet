"""Shared fixtures for the builder end-to-end tests: a fabricated tiny
2-session SSSD corpus (real 48 kHz 2-channel wavs, gzip jsonl manifests),
following the F5 recipe's ``dataset/tests/conftest.py`` fixture pattern
(``egs3/conversational/tts/dataset/tests/conftest.py``).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SOURCE_SR = 48000


def _write_wav(
    path: Path, freqs: list[float], duration_s: float, sr: int = SOURCE_SR
) -> None:
    """Write a synthetic multi-channel wav where channel i is a pure tone at
    ``freqs[i]`` Hz (kept inside pyin's 65-400 Hz search range so per-speaker
    measurement succeeds on this fixture)."""
    t = np.arange(int(round(duration_s * sr))) / sr
    data = np.stack([0.3 * np.sin(2 * math.pi * f * t) for f in freqs], axis=1).astype(
        np.float32
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


# Each session alternates two speakers, one per channel, with real
# (if short) sentences so words_per_sec is measurable.  sessB opens with a
# single-channel-only stretch (spk0 alone) so at least one window in the
# fixture has < 2 active speakers, exercising the TAC-drop path end to end.
_SESSIONS = {
    "sessA": {
        "num_channels": 2,
        "duration": 8.0,
        "freqs": [130.0, 210.0],
        "turns": [
            (0, "sessA_spk0", 0.5, 2.0, "hello there how are you"),
            (1, "sessA_spk1", 2.5, 4.0, "i am doing well thanks"),
            (0, "sessA_spk0", 4.5, 6.0, "that is good to hear"),
            (1, "sessA_spk1", 6.5, 7.8, "yes it is a nice day"),
        ],
    },
    "sessB": {
        "num_channels": 2,
        "duration": 12.0,
        "freqs": [150.0, 240.0],
        "turns": [
            (0, "sessB_spk0", 0.5, 2.0, "the weather today is lovely and warm"),
            (0, "sessB_spk0", 3.0, 4.5, "we should go outside soon"),
            (1, "sessB_spk1", 6.0, 7.5, "that sounds like a great idea"),
            (0, "sessB_spk0", 8.0, 9.5, "lets pack a small picnic basket"),
            (1, "sessB_spk1", 10.0, 11.5, "i will bring some cold drinks"),
        ],
    },
}


@pytest.fixture
def bagpiper_corpus(tmp_path):
    """Fabricate the tiny 2-session corpus tree under ``tmp_path/corpus``."""
    root = tmp_path / "corpus"
    recordings = []
    supervisions = []
    for session_id, spec in _SESSIONS.items():
        wav_name = f"{session_id}_mixed.wav"
        _write_wav(
            root / "original" / wav_name,
            spec["freqs"],
            spec["duration"],
        )
        recordings.append(
            {
                "id": session_id,
                "sources": [
                    {
                        "type": "file",
                        "channels": list(range(spec["num_channels"])),
                        # Bogus absolute prefix on purpose: the loader must
                        # remap onto dataset_root, not trust this path.
                        "source": f"/scratch/elsewhere/original/{wav_name}",
                    }
                ],
                "sampling_rate": SOURCE_SR,
                "num_samples": int(spec["duration"] * SOURCE_SR),
                "duration": spec["duration"],
                "channel_ids": list(range(spec["num_channels"])),
            }
        )
        for i, (channel, speaker, start, end, text) in enumerate(spec["turns"]):
            supervisions.append(
                {
                    "id": f"{session_id}_utt{i:04d}",
                    "recording_id": session_id,
                    "start": start,
                    "duration": round(end - start, 3),
                    "channel": channel,
                    "text": text,
                    "speaker": speaker,
                }
            )

    manifests = root / "lhotse_manifests_48"
    manifests.mkdir(parents=True)
    with gzip.open(manifests / "recordings.jsonl.gz", "wt", encoding="utf-8") as f:
        for rec in recordings:
            f.write(json.dumps(rec) + "\n")
    with gzip.open(manifests / "supervisions.jsonl.gz", "wt", encoding="utf-8") as f:
        for sup in supervisions:
            f.write(json.dumps(sup) + "\n")

    return {"root": root, "session_ids": sorted(_SESSIONS)}


@pytest.fixture
def tiny_builder_cfg():
    """A ``builder`` config block sized for the fixture's short audio (a few
    seconds), so windowing + pyin stay fast. Split ratios match the real
    ``conf/dataset.yaml`` (0.96/0.02/0.02): with only 2 sessions both land in
    train, which also exercises the empty-valid/empty-test output path."""
    return {
        "manifests_subdir": "lhotse_manifests_48",
        "audio_subdir": "original",
        "merge_gap": 0.5,
        "window_min": 3.0,
        "window_max": 6.0,
        "boundary_guard": 0.0,
        "tail_min": 1.0,
        "split_ratios": {"train": 0.96, "valid": 0.02, "test": 0.02},
        "source_sample_rate": SOURCE_SR,
        "target_sample_rate": 16000,
        "measure_cap_sec": 120.0,
        "seed": 0,
    }
