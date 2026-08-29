"""Lexical (paper) backchannel rule: normaliser, lexicon derivation, labeler,
and its plumbing through ``TurnTakingJudgeMetric`` (``bc_rule``/``tag``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from egs3.conversational.tts.src.metrics.backchannel_lexicon import (
    LexicalBackchannelLabeler,
    build_lexicon,
    default_lexicon_path,
    load_lexicon,
    normalize_phrase,
)
from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    ROLE_KEYS,
    TurnTakingJudge,
    TurnTakingJudgeMetric,
    apply_backchannel_proxy,
    label_rows,
)
from egs3.conversational.tts.tests.test_turn_taking_judge import (
    KeyedFakeVADBackend,
    _oracle_table,
    _unique_wav,
    _write_meta_scp,
    _write_window,
)

SR = 16000


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expect",
        [
            ("Mm-hmm.", "um-hum"),
            ("Uh-huh", "uh-huh"),
            ("uh huh", "uh-huh"),
            ("um-hum", "um-hum"),
            ("Yeah.", "yeah"),
            ("Okay, right.", "okay right"),
            ("Oh, I see.", "oh i see"),
            ("Hmm", "hm"),
            ("mhm", "um-hum"),
            ("[noise] right", "right"),
            ("[laughter-yeah]", ""),  # bracketed token is an annotation, not speech
            ("That's right!", "that's right"),
            ("  Yeah   yeah ", "yeah yeah"),
            ("", ""),
        ],
    )
    def test_whisper_and_switchboard_spellings_meet(self, raw, expect):
        assert normalize_phrase(raw) == expect

    def test_hesitations_are_kept_not_deleted(self):
        # whisper's EnglishTextNormalizer would delete these; ours must not
        for w in ("um", "uh", "hm", "um-hum", "uh-huh"):
            assert normalize_phrase(w) == w


class TestLexicon:
    def test_build_from_words_column_with_cutoff(self, tmp_path):
        rows = ["start,end,text,words,bc_label"]
        rows += ["0,1,[noise] right,\"['right']\",dialog-bc"] * 3
        rows += ["0,1,um-hum,\"['um-hum']\",dialog-bc"] * 5
        rows += ["0,1,oh yeah,\"['oh', 'yeah']\",dialog-bc"] * 2
        rows += ["0,1,that is very true,\"['that', 'is', 'very', 'true']\",dialog-bc"] * 9
        csv_path = tmp_path / "bc.csv"
        csv_path.write_text("\n".join(rows) + "\n")
        got = build_lexicon(csv_path, min_count=3, max_words=2)
        assert got == [("um-hum", 5), ("right", 3)]  # 3-word phrase and rare one out

    def test_committed_lexicon_loads_and_has_the_top_phrases(self):
        lex = load_lexicon(default_lexicon_path())
        assert {"yeah", "um-hum", "uh-huh", "right", "okay", "oh yeah", "i see"} <= lex
        assert all(len(p.split()) <= 2 for p in lex)
        assert len(lex) == 36

    def test_empty_lexicon_raises(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("# only a comment\n")
        with pytest.raises(ValueError):
            load_lexicon(f)


class SpanTranscriber:
    """Fake ASR keyed by the clip's (start, end) in seconds; counts calls."""

    def __init__(self, table, wav_len):
        self.table = {(round(s, 3), round(e, 3)): t for (s, e), t in table.items()}
        self.calls = []
        self.wav_len = wav_len

    def __call__(self, wav, sr):
        n = len(wav)
        # recover the span from the clip length + the marker sample we plant
        key = next(
            k for k in self.table if abs(int(round((k[1] - k[0]) * sr)) - n) <= 1
            and self.table[k] is not None
        )
        self.calls.append(key)
        return self.table[key]


class TestLabeler:
    def _mk(self, table, max_ipu_sec=3.0):
        tr = SpanTranscriber(table, None)
        return tr, LexicalBackchannelLabeler(
            tr, lexicon=["yeah", "um-hum", "oh yeah", "Right"], max_ipu_sec=max_ipu_sec
        )

    def test_phrase_during_other_floor_is_bc_regardless_of_duration(self):
        # channel 1 says a 2.0 s "yeah" while channel 0 holds the floor: BC
        # under the paper rule (the duration proxy would have said IPU)
        spans = [[(0.0, 5.0)], [(1.0, 3.0)]]
        tr, lab = self._mk({(0.0, 5.0): "long turn", (1.0, 3.0): "Yeah."})
        wavs = [np.zeros(SR * 6, np.float32), np.zeros(SR * 6, np.float32)]
        recs = lab.transcribe_spans(spans, wavs, SR)
        out = lab.label_records(recs)
        assert out[1] == [((1.0, 3.0), "BC")]
        assert out[0] == [((0.0, 5.0), "IPU")]
        assert recs[1][0]["normalized"] == "yeah" and recs[1][0]["is_bc"] is True
        assert apply_backchannel_proxy(spans, 6.0)[1] == [((1.0, 3.0), "IPU")]

    def test_non_lexicon_short_ipu_is_not_bc(self):
        spans = [[(0.0, 5.0)], [(1.0, 1.5)]]
        _, lab = self._mk({(0.0, 5.0): "x", (1.0, 1.5): "no way"})
        wavs = [np.zeros(SR * 6, np.float32)] * 2
        out = lab.label_records(lab.transcribe_spans(spans, wavs, SR))
        assert out[1] == [((1.0, 1.5), "IPU")]
        assert apply_backchannel_proxy(spans, 6.0)[1] == [((1.0, 1.5), "BC")]

    def test_phrase_after_taking_the_floor_is_not_bc(self):
        wavs = [np.zeros(SR * 6, np.float32)] * 2
        # other channel's floor (ended 1.0) is still the last floor at 2.0 -> BC
        spans = [[(0.0, 1.0)], [(2.0, 2.5)]]
        _, lab = self._mk({(0.0, 1.0): "hi", (2.0, 2.5): "yeah"})
        assert lab.label_records(lab.transcribe_spans(spans, wavs, SR))[1] == [
            ((2.0, 2.5), "BC")
        ]
        # channel 1 took the floor at 2.6; its later "yeah" is its own turn
        spans = [[(0.0, 1.0)], [(2.6, 4.0), (4.5, 5.0)]]
        _, lab = self._mk({(0.0, 1.0): "hi", (2.6, 4.0): "so", (4.5, 5.0): "yeah"})
        out = lab.label_records(lab.transcribe_spans(spans, wavs, SR))
        assert out[1] == [((2.6, 4.0), "IPU"), ((4.5, 5.0), "IPU")]

    def test_long_ipus_are_not_transcribed(self):
        spans = [[(0.0, 4.0)], [(0.5, 1.0)]]
        tr, lab = self._mk({(0.0, 4.0): None, (0.5, 1.0): "um-hum"}, max_ipu_sec=3.0)
        wavs = [np.zeros(SR * 5, np.float32)] * 2
        recs = lab.transcribe_spans(spans, wavs, SR)
        assert recs[0][0]["candidate"] is False and recs[0][0]["text"] is None
        assert tr.calls == [(0.5, 1.0)]
        assert lab.label_records(recs)[1] == [((0.5, 1.0), "BC")]

    def test_three_channels_all_ipu(self):
        spans = [[(0.0, 1.0)], [(0.2, 0.5)], [(0.6, 0.9)]]
        _, lab = self._mk({(0.0, 1.0): "a", (0.2, 0.5): "yeah", (0.6, 0.9): "yeah"})
        wavs = [np.zeros(SR * 2, np.float32)] * 3
        out = lab.label_records(lab.transcribe_spans(spans, wavs, SR))
        assert all(k == "IPU" for ch in out for _, k in ch)


class TestLabelRowsOverride:
    def test_labelled_override_changes_bc_chunks_only(self):
        spans = [[(0.2, 3.0)], [(1.0, 2.5)]]  # 1.5 s: IPU under the duration proxy
        base = label_rows("w", spans, 4.0)
        forced = [[((0.2, 3.0), "IPU")], [((1.0, 2.5), "BC")]]
        over = label_rows("w", spans, 4.0, labelled=forced)
        assert len(base) == len(over)
        labs_b = [r.split(",")[3] for r in base]
        labs_o = [r.split(",")[3] for r in over]
        assert "BC" in labs_o and "I" in labs_b
        assert base[0] == over[0]  # grid identical


class TestMetricPlumbing:
    def _window(self, tmp_path, vad):
        test_dir = tmp_path / "infer" / "valid"
        dur = 4.0
        g0 = vad.register(_unique_wav(1, int(dur * SR)), [(0.2, 3.0)])
        g1 = vad.register(_unique_wav(2, int(dur * SR)), [(1.0, 2.5)])
        _write_window(test_dir, "w1", dur, [g0, g1], seed=99)
        _write_meta_scp(test_dir, ["w1"])
        return test_dir, dur

    def test_lexical_tag_isolates_keys_dirs_and_shares_likelihoods(self, tmp_path):
        vad = KeyedFakeVADBackend()
        test_dir, dur = self._window(tmp_path, vad)
        forced = [[((0.2, 3.0), "IPU")], [((1.0, 2.5), "BC")]]
        table = _oracle_table(label_rows("w1", [[(0.2, 3.0)], [(1.0, 2.5)]], dur, labelled=forced))
        n_encode = {"n": 0}

        def encode(batch):
            n_encode["n"] += 1
            return np.stack([table[round(0.24 + 0.04 * 0, 2)]] * len(batch))

        class FixedTranscriber:
            def __init__(self):
                self.calls = 0

            def __call__(self, wav, sr):
                self.calls += 1
                return "Uh-huh." if len(wav) == int(1.5 * SR) else "a long turn here"

        tr = FixedTranscriber()
        judge = TurnTakingJudge(encode_fn=encode)
        data = {"meta": test_dir / "meta.scp"}
        out_root = tmp_path / "infer"

        # duration rule first (default), untagged
        s_dur = TurnTakingJudgeMetric(judge=judge, vad_backend=vad)(data, "valid", out_root)
        # 1.5 s "uh-huh" is over the 1.08 s cap: IPU under the duration proxy
        assert "judge_f1_macro" in s_dur and s_dur["judge_bc_proxy_count"] == 0

        # lexical rule, tagged
        m = TurnTakingJudgeMetric(
            judge=judge, vad_backend=vad, bc_rule="lexical", transcriber=tr, tag="lex"
        )
        s_lex = m(data, "valid", out_root)
        assert set(s_lex) == {
            (f"judge_lex_{k[len('judge_'):]}") for k in s_dur
        }
        assert s_lex["judge_lex_bc_proxy_count"] == 1
        d = out_root / "valid" / "scoring" / "turn_taking_judge"
        assert (d / "labels" / "w1.txt").exists() and (d / "labels_lex" / "w1.txt").exists()
        assert (d / "summary.json").exists() and (d / "summary_lex.json").exists()
        assert (d / "windows_lex.jsonl").exists() and (d / "confusion_lex.json").exists()
        assert len(list((d / "likelihoods").glob("*.txt"))) == 1
        lab_dur = [r.split(",")[3] for r in (d / "labels" / "w1.txt").read_text().splitlines()]
        lab_lex = [r.split(",")[3] for r in (d / "labels_lex" / "w1.txt").read_text().splitlines()]
        assert "BC" in lab_lex and "BC" not in lab_dur
        # transcript cache: both IPUs recorded, only the short one transcribed
        cache = json.loads((d / "bc_lexical_lex" / "w1.json").read_text())
        recs = cache["channels"]
        assert recs[1][0]["is_bc"] is True and recs[1][0]["normalized"] == "uh-huh"
        assert recs[0][0]["candidate"] is True  # 2.8 s <= 3.0
        calls_after_first = tr.calls
        # second run: likelihoods AND transcripts come from cache
        n_before = n_encode["n"]
        m(data, "valid", out_root)
        assert n_encode["n"] == n_before and tr.calls == calls_after_first
        # summary.json of the duration rule untouched by the lexical run
        assert json.loads((d / "summary.json").read_text())["judge_bc_proxy_count"] == 0
        assert all(k in s_lex for k in (f"judge_lex_{r[len('judge_'):]}" for r in ROLE_KEYS))

    def test_bad_rule_rejected(self):
        with pytest.raises(ValueError):
            TurnTakingJudgeMetric(judge=TurnTakingJudge(encode_fn=lambda b: b), bc_rule="asr")

    def test_transcript_cache_invalidated_by_new_spans(self, tmp_path):
        vad = KeyedFakeVADBackend()
        test_dir, dur = self._window(tmp_path, vad)
        d = tmp_path / "infer" / "valid" / "scoring" / "turn_taking_judge"
        (d / "bc_lexical_lex").mkdir(parents=True)
        (d / "bc_lexical_lex" / "w1.json").write_text(
            json.dumps({"spans": [[[0.0, 9.0]], []], "channels": [[], []]})
        )
        calls = {"n": 0}

        def tr(wav, sr):
            calls["n"] += 1
            return "yeah"

        table = _oracle_table(label_rows("w1", [[(0.2, 3.0)], [(1.0, 2.5)]], dur))
        judge = TurnTakingJudge(encode_fn=lambda b: np.stack([table[0.24]] * len(b)))
        TurnTakingJudgeMetric(
            judge=judge, vad_backend=vad, bc_rule="lexical", transcriber=tr, tag="lex"
        )({"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer")
        assert calls["n"] == 2  # stale cache ignored, both IPUs re-transcribed
