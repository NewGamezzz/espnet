"""DNSMOS session blocklist: JSON parsing, key mapping, dataset filtering."""

import json

import pytest

from egs3.conversational.tts.dataset.preprocessing import blocklist
from egs3.conversational.tts.dataset.preprocessing.sessions import SessionRecord
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)
from .test_dataset import corpus, make_dataset  # noqa: F401  (fixture)


def test_channel_key_to_session_strips_known_suffixes():
    assert blocklist.channel_key_to_session("abc-ch0") == "abc"
    assert blocklist.channel_key_to_session("abc-ch1") == "abc"
    assert blocklist.channel_key_to_session("fe_03_00001-A") == "fe_03_00001"
    assert blocklist.channel_key_to_session("fe_03_00001-B") == "fe_03_00001"
    with pytest.raises(ValueError, match="suffix"):
        blocklist.channel_key_to_session("fe_03_00001")


def test_load_blocked_sessions_unions_files(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(
            {
                "dataset": "candor",
                "drop": {"x-ch0": ["dnsmos_ovrl=1.8"], "y-ch1": ["dnsmos_ovrl=2.1"]},
            }
        )
    )
    b.write_text(
        json.dumps({"dataset": "fisher", "drop": {"fe_03_00004-A": ["min=2.4"]}})
    )
    assert blocklist.load_blocked_sessions(a) == {"x", "y"}
    assert blocklist.load_blocked_sessions(str(a)) == {"x", "y"}
    assert blocklist.load_blocked_sessions([a, b]) == {"x", "y", "fe_03_00004"}


def test_load_blocked_sessions_requires_drop_object(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"dataset": "candor"}))
    with pytest.raises(ValueError, match="drop"):
        blocklist.load_blocked_sessions(p)


def _sess(sid, dur=3600.0):
    return SessionRecord(
        sid, f"{sid}.flac", 2, 24000, dur, (Turn(0, "s", "hi", 0.0, 1.0),)
    )


def test_apply_blocklist_drops_and_reports(caplog):
    sessions = [_sess("a"), _sess("b", 1800.0), _sess("c")]
    with caplog.at_level("INFO"):
        kept = blocklist.apply_session_blocklist(sessions, {"b", "zzz"}, source="t")
    assert [s.session_id for s in kept] == ["a", "c"]
    # ids from other splits ("zzz") are expected and do not raise
    assert "kept 2 / dropped 1 sessions (0.5 h dropped)" in caplog.text


def test_apply_blocklist_zero_overlap_raises():
    with pytest.raises(ValueError, match="none of the 2"):
        blocklist.apply_session_blocklist([_sess("a")], {"y", "zzz"}, source="t")
    # empty blocklist is a no-op, not an error
    assert len(blocklist.apply_session_blocklist([_sess("a")], set(), source="t")) == 1


def test_dataset_kwarg_filters_sessions(corpus, tmp_path):  # noqa: F811
    ds_all = make_dataset(corpus)
    sid = ds_all.sessions[0].session_id
    bl = tmp_path / "bl.json"
    bl.write_text(json.dumps({"drop": {f"{sid}-ch0": ["dnsmos_ovrl=1.0"]}}))
    ds = make_dataset(corpus, session_blocklist=bl)
    assert len(ds.sessions) == len(ds_all.sessions) - 1
    assert sid not in {s.session_id for s in ds.sessions}
    assert len(ds) < len(ds_all)
