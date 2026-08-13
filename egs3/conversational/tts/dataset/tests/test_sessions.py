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

    def test_write_is_atomic(self, tmp_path):
        path = tmp_path / "sessions.jsonl"
        write_session_manifest(path, [_session()])
        assert not path.with_suffix(".jsonl.tmp").exists()

    def test_empty_manifest_raises(self, tmp_path):
        (tmp_path / "empty.jsonl").write_text("")
        with pytest.raises(RuntimeError):
            read_session_manifest(tmp_path / "empty.jsonl")
