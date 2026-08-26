"""Tests for ``local/mooncast_collect.py``, the MoonCast run verifier.

The runner already writes ``<window_id>.wav``, so unlike the MOSS-TTSD
collector this script renames nothing.  Its whole job is to refuse a table
that is quietly incomplete: every dialogue present exactly once, audio on
disk for each, and any runaway turn flagged rather than kept as a normal
row.  MoonCast's own per-turn cap is 200 seconds, which nothing on this set
could reach, so the threshold that matters here is ours.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mooncast_collect",
    Path(__file__).resolve().parents[1] / "local" / "mooncast_collect.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def record(wid, *, status="ok", frames=(100, 100), duration=16.0, wall=30.0):
    return {
        "window_id": wid,
        "status": status,
        "seed": 1,
        "num_turns_in": len(frames),
        "num_turns_generated": len(frames),
        "duration_sec": duration,
        "wall_sec": wall,
        "turns": [
            {"role": "0", "samples": f * 480, "frames": f, "start": 0, "end": 1}
            for f in frames
        ],
    }


def write_run(tmp_path, shards, wavs=True):
    """A run directory of ``{shard_name: [records]}``, with its wavs."""
    for name, records in shards.items():
        shard = tmp_path / name
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        if wavs:
            for r in records:
                if r["status"] == "ok":
                    (shard / f"{r['window_id']}.wav").write_bytes(b"RIFF")
    return tmp_path


class TestVerify:
    def test_a_complete_run_reports_its_counts(self, tmp_path):
        run = write_run(tmp_path, {"00": [record("d1")], "01": [record("d2")]})
        report = mod.verify(run, ["d1", "d2"])
        assert report["collected"] == 2
        assert report["runaway"] == []
        assert report["total_wall_sec"] == pytest.approx(60.0)
        assert report["duration_sec"]["mean"] == pytest.approx(16.0)

    def test_a_missing_dialogue_is_an_error(self, tmp_path):
        run = write_run(tmp_path, {"00": [record("d1")]})
        with pytest.raises(ValueError, match="d2"):
            mod.verify(run, ["d1", "d2"])

    def test_a_failed_row_is_an_error_naming_it(self, tmp_path):
        run = write_run(tmp_path, {"00": [record("d1", status="failed")]})
        with pytest.raises(ValueError, match="d1"):
            mod.verify(run, ["d1"])

    def test_a_missing_wav_is_an_error(self, tmp_path):
        run = write_run(tmp_path, {"00": [record("d1")]}, wavs=False)
        with pytest.raises(FileNotFoundError):
            mod.verify(run, ["d1"])

    def test_a_retry_supersedes_the_attempt_it_repeats(self, tmp_path):
        # The runner APPENDS, so a resumed shard leaves both attempts on
        # disk.  The last one is the run; the first is history.
        run = write_run(tmp_path, {"00": [record("d1", status="failed"), record("d1")]})
        report = mod.verify(run, ["d1"])
        assert report["collected"] == 1

    def test_the_same_dialogue_in_two_shards_is_an_error(self, tmp_path):
        # Appending inside one shard is a retry; the same id in two shards
        # means the input was split wrong, and one of the two wavs is
        # silently unused.
        run = write_run(tmp_path, {"00": [record("d1")], "01": [record("d1")]})
        with pytest.raises(ValueError, match="two shards"):
            mod.verify(run, ["d1"])

    def test_a_runaway_turn_is_flagged(self, tmp_path):
        # Their cap is 200 s on every turn, so a turn that never stopped
        # hides inside a plausible-looking total duration rather than being
        # clipped.  30 s of one turn is ours, not theirs.
        run = write_run(tmp_path, {"00": [record("d1", frames=(100, 1500))]})
        report = mod.verify(run, ["d1"])
        assert report["runaway"] == [("d1", 1, 1500)]

    def test_a_long_but_plausible_turn_is_not_flagged(self, tmp_path):
        run = write_run(tmp_path, {"00": [record("d1", frames=(100, 1499))]})
        assert mod.verify(run, ["d1"])["runaway"] == []

    def test_the_threshold_is_thirty_seconds_at_fifty_hertz(self):
        assert mod.RUNAWAY_FRAMES == 1500
        assert mod.FRAMES_PER_SEC == 50

    def test_a_run_directory_without_records_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mod.verify(tmp_path, ["d1"])

    def test_wavs_are_gathered_into_one_directory_for_ingest(self, tmp_path):
        # Shards run as separate jobs into separate directories, but the
        # ingest's wav_dir is a single path.
        run = write_run(tmp_path / "run", {"00": [record("d1")], "01": [record("d2")]})
        out = tmp_path / "collected"
        report = mod.verify(run, ["d1", "d2"], out_dir=out)
        assert sorted(p.name for p in out.glob("*.wav")) == ["d1.wav", "d2.wav"]
        assert report["collected"] == 2
