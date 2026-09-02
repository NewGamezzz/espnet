"""AMIBuilder: lexical K strata, channel subsets, session/window manifests and
the window report - on a fabricated 4-headset meeting (no corpus access)."""
import json
import string
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.ami_builder import (
    AMIBuilder,
    lexical_active_channels,
    stratify_window,
)
from egs3.conversational.tts.dataset.preprocessing.ami import transcode_meeting
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.windows import WindowRecord, from_json

NITE = 'xmlns:nite="http://nite.sourceforge.net/"'


def _rec(turns, channels=None):
    return WindowRecord(
        window_id="m_w0",
        session_id="m",
        audio_relpath="ami_flac/m.flac",
        num_channels=4,
        sample_rate=24000,
        t0=0.0,
        t1=20.0,
        turns=tuple(turns),
        channels=channels,
    )


class TestStratify:
    def test_lexical_rule_counts_three_word_turns_only(self):
        turns = [
            Turn(0, "a", "one two three", 0.5, 3.0),
            Turn(1, "b", "yeah", 3.2, 3.6),  # sub-lexical
            Turn(2, "c", "four five six seven", 4.0, 8.0),
            Turn(3, "d", "mm hmm", 8.5, 9.0),  # sub-lexical
        ]
        assert lexical_active_channels(turns, min_words=3) == (0, 2)

    def test_stratify_drops_silent_participants_turns_and_sets_channels(self):
        r = _rec(
            [
                Turn(0, "a", "one two three", 0.5, 3.0),
                Turn(1, "b", "yeah", 3.2, 3.6),
                Turn(2, "c", "four five six seven", 4.0, 8.0),
            ]
        )
        out = stratify_window(r, min_words=3)
        assert out.channels == (0, 2)
        assert [t.channel for t in out.turns] == [0, 2]
        assert out.num_active_speakers == 2 and out.num_rows == 2

    def test_stratify_keeps_single_speaker_as_pool_only(self):
        r = _rec(
            [
                Turn(0, "a", "one two three", 0.5, 3.0),
                Turn(1, "b", "hm", 3.5, 3.7),
                Turn(0, "a", "four five six", 4.0, 6.0),
            ]
        )
        out = stratify_window(r, min_words=3)
        assert out.channels == (0,) and out.num_rows == 1
        assert [t.channel for t in out.turns] == [0, 0]  # the fragment is gone

    def test_stratify_returns_none_without_lexical_speaker(self):
        r = _rec([Turn(0, "a", "yeah", 0.5, 0.9), Turn(1, "b", "hm", 5.0, 5.2)])
        assert stratify_window(r, min_words=3) is None


@pytest.fixture
def base_vocab_file(tmp_path) -> Path:
    tokens = (
        ["<blank>", "<unk>", "<space>"]
        + list(string.ascii_lowercase)
        + [".", ",", "?", "!", "'", "<sos/eos>"]
    )
    p = tmp_path / "base_vocab.txt"
    p.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return p


def _make_corpus(tmp_path: Path, seconds: float = 120.0) -> Path:
    """A and B alternate 3-word turns every 4 s; C says one 3-word turn at
    50 s; D only backchannels ("yeah") every 10 s."""
    root = tmp_path / "ami"
    mid = "ES2004a"
    ann = root / "annotations"
    (ann / "words").mkdir(parents=True)
    (ann / "corpusResources").mkdir(parents=True)
    (ann / "corpusResources" / "meetings.xml").write_text(
        f"""<?xml version="1.0"?>
<nite:root {NITE}><meeting observation="{mid}">
<speaker channel="0" nxt_agent="A" global_name="MEO015"/>
<speaker channel="1" nxt_agent="B" global_name="FEE013"/>
<speaker channel="2" nxt_agent="C" global_name="MEE014"/>
<speaker channel="3" nxt_agent="D" global_name="FEE016"/>
</meeting></nite:root>"""
    )

    def turns_for(ch):
        out = []
        if ch in (0, 1):
            t = 1.0 + 4.0 * ch  # A at 1, 9, 17...; B at 5, 13, 21...
            while t < seconds - 5:
                out.append((t, t + 2.4, "one two three"))
                t += 8.0
        elif ch == 2:
            out.append((50.0, 52.4, "four five six"))
        else:
            t = 3.4
            while t < seconds - 5:
                out.append((t, t + 0.4, "yeah"))
                t += 10.0
        return out

    for ch, agent in enumerate("ABCD"):
        rows = []
        i = 0
        for start, end, text in turns_for(ch):
            toks = text.split()
            step = (end - start) / len(toks)
            for j, tok in enumerate(toks):
                rows.append(
                    f'<w nite:id="w{i}" starttime="{start + j * step:.2f}" '
                    f'endtime="{start + (j + 1) * step:.2f}">{tok}</w>'
                )
                i += 1
        (ann / "words" / f"{mid}.{agent}.words.xml").write_text(
            f'<?xml version="1.0"?><nite:root {NITE}>' + "".join(rows) + "</nite:root>"
        )
    audio = root / "amicorpus" / mid / "audio"
    audio.mkdir(parents=True)
    sr = 16000
    t = np.arange(int(seconds * sr)) / sr
    for ch in range(4):
        sf.write(
            str(audio / f"{mid}.Headset-{ch}.wav"),
            (0.2 * np.sin(2 * np.pi * 200 * (ch + 1) * t)).astype("float32"),
            sr,
        )
    transcode_meeting(root, mid, root / "ami_flac")
    return root


@pytest.fixture
def short_windows(monkeypatch):
    # The synthetic corpus is sparse; the production band (15-45 s, tail 10 s)
    # stays in config.yaml, the test only needs windows to exist.
    from egs3.conversational.tts.dataset import ami_builder

    monkeypatch.setitem(ami_builder._CFG, "window_min", 8.0)
    monkeypatch.setitem(ami_builder._CFG, "tail_min", 5.0)


class TestBuild:
    def test_build_writes_manifests_and_report(self, tmp_path, base_vocab_file, short_windows):
        root = _make_corpus(tmp_path)
        recipe = tmp_path / "recipe"
        AMIBuilder().build(
            recipe, dataset_root=root, base_vocab_path=base_vocab_file, meetings=["ES2004a"]
        )
        lines = (recipe / "data/manifest/ami_test.jsonl").read_text().splitlines()
        windows = [from_json(json.loads(l)) for l in lines]
        assert len(windows) >= 3, "synthetic corpus should yield several windows at window_min 8"
        for w in windows:
            assert 5.0 <= w.duration <= 45.0 + 1e-6
            assert w.num_channels == 4 and w.channels is not None
            assert w.num_rows == w.num_active_speakers >= 1
            assert 3 not in w.channels  # D never has a 3-word turn
            assert w.sample_rate == 24000 and w.audio_relpath == "ami_flac/ES2004a.flac"
        assert any(w.num_rows == 3 for w in windows)  # the window around C's turn
        sessions = [
            json.loads(l)
            for l in (recipe / "data/manifest/ami_test_sessions.jsonl").read_text().splitlines()
        ]
        assert sessions[0]["session_id"] == "ES2004a"
        assert sessions[0]["speakers"] == {
            "0": "MEO015", "1": "FEE013", "2": "MEE014", "3": "FEE016",
        }
        assert any(t["channel"] == 3 for t in sessions[0]["turns"])  # full annotation kept
        report = json.loads((recipe / "exp/ami/window_report.json").read_text())
        assert report["meetings"] == ["ES2004a"]
        assert set(report["per_k"]) <= {"1", "2", "3", "4"}
        assert sum(v["windows"] for v in report["per_k"].values()) == len(windows)
        assert report["dropped"]["no_lexical_speaker"] >= 0

    def test_build_is_deterministic(self, tmp_path, base_vocab_file, short_windows):
        root = _make_corpus(tmp_path)
        a, b = tmp_path / "r1", tmp_path / "r2"
        for r in (a, b):
            AMIBuilder().build(
                r, dataset_root=root, base_vocab_path=base_vocab_file, meetings=["ES2004a"]
            )
        assert (a / "data/manifest/ami_test.jsonl").read_bytes() == (
            b / "data/manifest/ami_test.jsonl"
        ).read_bytes()
