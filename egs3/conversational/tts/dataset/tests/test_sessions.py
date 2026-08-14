import json
from pathlib import Path

import pytest

from ..preprocessing.sessions import (
    SessionRecord,
    from_json,
    read_session_manifest,
    to_json,
    write_session_manifest,
)
from ..preprocessing.sssd import Turn
from ..preprocessing.windows import WindowRecord
from ..preprocessing.windows import to_json as window_to_json


def _session(**kw):
    base = dict(
        session_id="sess_a",
        audio_relpath="original/sess_a_mixed.flac",
        num_channels=2,
        sample_rate=48000,
        duration=30.123456789,  # deliberately not 6-dp-round
        turns=(Turn(channel=0, speaker="s0", text="hi there", start=0.5, end=2.0),),
    )
    base.update(kw)
    return SessionRecord(**base)


class TestRoundTrip:
    def test_json_roundtrip_is_exact(self):
        s = _session(exclusion_spans=((3.0, 4.5),))
        assert from_json(json.loads(json.dumps(to_json(s)))) == s

    def test_floats_are_not_rounded(self):
        # Parity depends on unrounded times: json round-trips floats exactly.
        s = _session(turns=(Turn(0, "s0", "x", 0.1000000001, 1.9999999999),))
        d = to_json(s)
        assert d["turns"][0]["start"] == 0.1000000001
        assert d["duration"] == 30.123456789

    def test_atomic_preserves_window_id(self):
        s = _session(atomic=True, window_id="libritts_100_a_utt1")
        r = from_json(to_json(s))
        assert r.atomic and r.window_id == "libritts_100_a_utt1"


class TestManifestIO:
    def test_write_read_roundtrip(self, tmp_path):
        recs = [_session(), _session(session_id="sess_b")]
        n = write_session_manifest(tmp_path / "m" / "sessions.jsonl", recs)
        assert n == 2
        assert read_session_manifest(tmp_path / "m" / "sessions.jsonl") == recs

    def test_write_publishes_via_os_replace(self, tmp_path, monkeypatch):
        """Verify the atomic-write contract: data goes to tmp, then os.replace publishes."""
        import os as os_module

        path = tmp_path / "sessions.jsonl"
        calls = []
        real_replace = os_module.replace

        def spy_replace(src, dst):
            calls.append((src, dst))
            # Call the real os.replace after recording the call
            real_replace(src, dst)

        monkeypatch.setattr(
            "egs3.conversational.tts.dataset.preprocessing.sessions.os.replace",
            spy_replace,
        )
        write_session_manifest(path, [_session()])

        # Verify os.replace was called exactly once with correct src/dst
        assert len(calls) == 1
        src, dst = calls[0]
        assert src == path.with_suffix(".jsonl.tmp")
        assert dst == path
        # Verify tmp file is gone after publish
        assert not path.with_suffix(".jsonl.tmp").exists()
        # Verify final file exists and is readable
        assert path.exists()

    def test_write_crashes_safely_before_publish(self, tmp_path, monkeypatch):
        """Verify crash-safety: tmp file has complete data even if os.replace fails."""
        path = tmp_path / "sessions.jsonl"
        tmp_path_expected = path.with_suffix(".jsonl.tmp")
        records = [_session(), _session(session_id="sess_b")]

        def crash_on_replace(src, dst):
            raise OSError("Simulated crash during publish")

        monkeypatch.setattr(
            "egs3.conversational.tts.dataset.preprocessing.sessions.os.replace",
            crash_on_replace,
        )

        # Call write_session_manifest; expect it to fail during publish
        with pytest.raises(OSError, match="Simulated crash during publish"):
            write_session_manifest(path, records)

        # Final file must not exist (publish failed)
        assert not path.exists()
        # Tmp file must exist with all data intact (written before publish)
        assert tmp_path_expected.exists()
        # Tmp file must have complete, parseable JSONL
        parsed = read_session_manifest(tmp_path_expected)
        assert parsed == records

    def test_empty_manifest_raises(self, tmp_path):
        (tmp_path / "empty.jsonl").write_text("")
        with pytest.raises(RuntimeError):
            read_session_manifest(tmp_path / "empty.jsonl")


class TestRejectsRetiredWindowManifest:
    """A retired WINDOW-manifest line has every key from_json requires, but
    its ``duration`` is the WINDOW duration while turns carry absolute
    session seconds - silently accepted, it would train on ~the first
    window per session. from_json must reject it instead."""

    def _window_manifest_line(self) -> dict:
        rec = WindowRecord(
            window_id="sess_a_w00000",
            session_id="sess_a",
            audio_relpath="original/sess_a_mixed.flac",
            num_channels=2,
            sample_rate=48000,
            t0=0.0,
            t1=8.5,
            turns=(Turn(channel=0, speaker="s0", text="hi there", start=0.5, end=2.0),),
        )
        return window_to_json(rec)

    def test_from_json_rejects_retired_window_manifest(self):
        d = self._window_manifest_line()
        with pytest.raises(ValueError, match="retired WINDOW manifest"):
            from_json(d)

    def test_read_session_manifest_rejects_retired_window_manifest(self, tmp_path):
        path = tmp_path / "fisher_train.jsonl"
        path.write_text(json.dumps(self._window_manifest_line()) + "\n")
        with pytest.raises(ValueError, match="retired WINDOW manifest"):
            read_session_manifest(path)
