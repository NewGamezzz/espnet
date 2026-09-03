"""local/build_ami_longform.py: one AMI meeting -> one external-manifest
dialogue (first in-band turn per participant as the prompt, the rest as the
script, masked full-length ground truth), on a fabricated 4-channel meeting."""
import json
import string
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.local.build_ami_longform import (
    build_longform,
    pick_prompt_turn,
)
from egs3.conversational.tts.local.build_zipvoice_dialog_testset import active_rms_db
from egs3.conversational.tts.src.external_testset import load_external_manifest

SR = 24000


def _turns():
    # A: "yeah" (sub-lexical), then a 4 s lexical turn (prompt), then more
    # B: an 0.8 s lexical fragment (too short), then a 3 s turn (prompt)
    # C: a 12 s turn (too long), then a 3 s turn overlapped by D (not solo),
    #    then a 2.5 s solo turn (prompt)
    # D: one 3 s turn (prompt) - D's only turn, so D has no script left
    return [
        Turn(0, "MEO015", "yeah", 0.5, 0.9),
        Turn(0, "MEO015", "one two three four", 2.0, 6.0),
        Turn(1, "FEE013", "hm okay right", 7.0, 7.8),
        Turn(1, "FEE013", "five six seven", 9.0, 12.0),
        Turn(2, "MEE014", "a long turn " * 5, 13.0, 25.0),
        Turn(2, "MEE014", "eight nine ten", 26.0, 29.0),
        Turn(3, "FEE016", "eleven twelve thirteen", 28.5, 31.5),
        Turn(2, "MEE014", "fourteen fifteen sixteen", 33.0, 35.5),
        Turn(0, "MEO015", "seventeen eighteen nineteen", 37.0, 40.0),
        Turn(1, "FEE013", "twenty twenty-one twenty-two", 41.0, 44.0),
        Turn(2, "MEE014", "twenty-three twenty-four twenty-five", 45.0, 48.0),
    ]


class TestPickPrompt:
    def test_first_lexical_in_band_solo_turn(self):
        turns = _turns()
        kw = dict(min_words=3, band=(2.0, 10.0), solo_guard=0.3)
        assert pick_prompt_turn(turns, 0, **kw) == (turns[1], "gated_solo")
        assert pick_prompt_turn(turns, 1, **kw) == (turns[3], "gated_solo")
        assert pick_prompt_turn(turns, 2, **kw) == (turns[7], "gated_solo")
        # D's only turn overlaps C: no solo tier, so the in-band tier keeps D
        assert pick_prompt_turn(turns, 3, **kw) == (turns[6], "in_band")

    def test_excluded_span_falls_to_the_next_gated_turn_then_to_solo(self):
        turns = _turns()
        kw = dict(min_words=3, band=(2.0, 10.0), solo_guard=0.3)
        assert pick_prompt_turn(turns, 0, excluded=frozenset({(0, 2.0, 6.0)}), **kw) == (turns[8], "gated_solo")
        both = frozenset({(0, 2.0, 6.0), (0, 37.0, 40.0)})
        assert pick_prompt_turn(turns, 0, excluded=both, **kw) == (turns[1], "solo")

    def test_no_lexical_in_band_turn_drops_the_participant(self):
        turns = [Turn(0, "a", "yeah", 0.5, 0.9), Turn(0, "a", "a very long turn " * 6, 2.0, 20.0)]
        assert pick_prompt_turn(turns, 0, min_words=3, band=(2.0, 10.0), solo_guard=0.3) is None


def _corpus(tmp_path: Path, seconds: float = 50.0):
    root = tmp_path / "ami"
    (root / "ami_flac").mkdir(parents=True)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.stack([0.2 * np.sin(2 * np.pi * 200 * (c + 1) * t) for c in range(4)], axis=1).astype("float32")
    sf.write(str(root / "ami_flac" / "ES2004a.flac"), x, SR, subtype="PCM_16", format="FLAC")
    sessions = tmp_path / "sessions.jsonl"
    line = {
        "session_id": "ES2004a", "audio_relpath": "ami_flac/ES2004a.flac", "sample_rate": SR,
        "num_channels": 4, "duration": seconds,
        "speakers": {"0": "MEO015", "1": "FEE013", "2": "MEE014", "3": "FEE016"},
        "turns": [{"channel": tr.channel, "speaker": tr.speaker, "text": tr.text,
                   "start": tr.start, "end": tr.end} for tr in _turns()],
    }
    sessions.write_text(json.dumps(line) + "\n")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["<blank>", "<unk>", "<space>"] + list(string.ascii_lowercase)
                               + [".", ",", "?", "!", "'", "-", "<sos/eos>", "<turn>", "<OTHER>"]) + "\n")
    return root, sessions, vocab


class TestBuild:
    def test_build_one_meeting(self, tmp_path):
        root, sessions, vocab = _corpus(tmp_path)
        out = tmp_path / "ami-longform"
        meta = build_longform(sessions=sessions, dataset_root=root, out_dir=out, exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=None)
        rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        r = rows[0]
        # D's only turn is its prompt (in_band tier) -> nothing left to say -> D dropped; K = 3
        assert r["window_id"] == "ES2004a" and r["num_channels"] == 3
        assert r["source_channels"] == [0, 1, 2]
        assert [c["prompt_text"] for c in r["channels"]] == [
            "one two three four", "five six seven", "fourteen fifteen sixteen"]
        assert [c["prompt_tier"] for c in r["channels"]] == ["gated_solo"] * 3
        texts = [t["text"] for t in r["turns"]]
        assert "one two three four" not in texts and "five six seven" not in texts
        assert "fourteen fifteen sixteen" not in texts
        assert "eleven twelve thirteen" not in texts  # D's turn is gone with D
        assert texts[0] == "yeah" and texts[-1] == "twenty-three twenty-four twenty-five"
        assert [t["channel"] for t in r["turns"]][:3] == [0, 1, 2]
        # prompt wav: mono, native rate, normalized, right length
        wav, sr = sf.read(str(out / r["channels"][0]["prompt_wav"]))
        assert wav.ndim == 1 and sr == SR and abs(len(wav) / sr - 4.0) < 0.01
        assert abs(active_rms_db(wav, sr) - (-23.0)) < 0.5
        # gt: full length, masked outside the channel's script turns (prompt turn masked too)
        gt, sr = sf.read(str(out / r["channels"][0]["gt_wav"]))
        assert gt.ndim == 1 and abs(len(gt) / sr - 50.0) < 0.01
        assert float(np.abs(gt[int(3.0 * SR):int(5.0 * SR)]).max()) == 0.0   # prompt turn masked
        assert float(np.abs(gt[int(0.6 * SR):int(0.8 * SR)]).max()) > 0.0    # "yeah" kept
        assert float(np.abs(gt[int(38.0 * SR):int(39.0 * SR)]).max()) > 0.0  # later turn kept
        assert float(np.abs(gt[int(20.0 * SR):int(21.0 * SR)]).max()) == 0.0  # C speaking, A silent
        assert meta["n_meetings"] == 1 and meta["meetings"]["ES2004a"]["num_channels"] == 3
        assert meta["meetings"]["ES2004a"]["dropped_channels"] == [3]
        assert meta["meetings"]["ES2004a"]["prompt_tiers"] == {"0": "gated_solo", "1": "gated_solo", "2": "gated_solo"}
        # loads through the real external-manifest loader
        recs = load_external_manifest(out / "manifest.jsonl", vocab)
        assert recs[0].num_channels == 3 and recs[0].gt_paths is not None
        assert abs(recs[0].gt_duration_sec - 50.0) < 0.01
        assert len(recs[0].turns) == len(r["turns"])

    def test_meeting_subset_and_determinism(self, tmp_path):
        root, sessions, vocab = _corpus(tmp_path)
        a = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "a", exclude_spans=None,
                           min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                           normalize_db=-23.0, meetings=["ES2004a"])
        b = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "b", exclude_spans=None,
                           min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                           normalize_db=-23.0, meetings=["ES2004a"])
        assert (tmp_path / "a" / "manifest.jsonl").read_bytes() == (tmp_path / "b" / "manifest.jsonl").read_bytes()
        none = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "c", exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=["ES2014a"])
        assert none["n_meetings"] == 0
