"""Tests for ``local/moss_ttsd_collect.py``: MOSS-TTSD writes
``{line_no:06d}.wav``, so the mapping back to our dialogue ids runs through
the ``window_id`` their ``output.jsonl`` copies through from our input.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_SPEC = importlib.util.spec_from_file_location(
    "moss_ttsd_collect",
    Path(__file__).resolve().parents[1] / "local" / "moss_ttsd_collect.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _wav(path: Path, seconds: float, sr: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * sr))
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr)
    return path


def _run_dir(tmp_path: Path, records: list[dict], shard: str = "shard_00") -> Path:
    run = tmp_path / "run"
    (run / shard).mkdir(parents=True, exist_ok=True)
    (run / shard / "output.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return run


def test_wavs_are_renamed_to_our_dialogue_ids(tmp_path):
    audio = _wav(tmp_path / "raw" / "000000.wav", 3.0)
    run = _run_dir(
        tmp_path,
        [{"window_id": "d2", "output_audio": str(audio), "duration": 3.0}],
    )
    out = tmp_path / "collected"
    report = mod.collect(run, out, expected_ids=["d2"])
    assert report["collected"] == 1
    assert report["runaway"] == []
    assert (out / "d2.wav").is_file()
    assert sf.info(str(out / "d2.wav")).duration == pytest.approx(3.0, abs=0.01)


def test_records_from_every_shard_are_collected(tmp_path):
    a = _wav(tmp_path / "raw" / "a.wav", 1.0)
    b = _wav(tmp_path / "raw" / "b.wav", 1.0)
    run = _run_dir(
        tmp_path, [{"window_id": "d1", "output_audio": str(a), "duration": 1.0}]
    )
    (run / "shard_01").mkdir()
    (run / "shard_01" / "output.jsonl").write_text(
        json.dumps({"window_id": "d2", "output_audio": str(b), "duration": 1.0}) + "\n",
        encoding="utf-8",
    )
    report = mod.collect(run, tmp_path / "collected", expected_ids=["d1", "d2"])
    assert report["collected"] == 2


def test_a_dropped_row_is_an_error_naming_the_id(tmp_path):
    # Their inference loop SKIPS a failed sample with a warning, so a short
    # table is the default failure mode and has to be caught here.
    audio = _wav(tmp_path / "raw" / "000000.wav", 3.0)
    run = _run_dir(
        tmp_path,
        [{"window_id": "d1", "output_audio": str(audio), "duration": 3.0}],
    )
    with pytest.raises(ValueError, match="d2"):
        mod.collect(run, tmp_path / "collected", expected_ids=["d1", "d2"])


def test_a_null_output_audio_is_an_error(tmp_path):
    run = _run_dir(
        tmp_path, [{"window_id": "d1", "output_audio": None, "duration": 0.0}]
    )
    with pytest.raises(ValueError, match="d1"):
        mod.collect(run, tmp_path / "collected", expected_ids=["d1"])


def test_a_duplicated_window_id_is_an_error(tmp_path):
    audio = _wav(tmp_path / "raw" / "000000.wav", 1.0)
    run = _run_dir(
        tmp_path,
        [
            {"window_id": "d1", "output_audio": str(audio), "duration": 1.0},
            {"window_id": "d1", "output_audio": str(audio), "duration": 1.0},
        ],
    )
    with pytest.raises(ValueError, match="d1"):
        mod.collect(run, tmp_path / "collected", expected_ids=["d1"])


def test_a_generation_near_the_token_cap_is_flagged_not_dropped(tmp_path):
    # 8192 tokens at 12.5 tokens/s is about 655 s, against a set whose
    # dialogues average roughly 24 s: an AR model looping is a finding.
    audio = _wav(tmp_path / "raw" / "000000.wav", 0.1)
    run = _run_dir(
        tmp_path,
        [{"window_id": "d1", "output_audio": str(audio), "duration": 650.0}],
    )
    report = mod.collect(run, tmp_path / "collected", expected_ids=["d1"])
    assert report["runaway"] == ["d1"]
    assert report["collected"] == 1
    assert (tmp_path / "collected" / "d1.wav").is_file()
