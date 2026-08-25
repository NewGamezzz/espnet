"""``TurnTakingJudgeMetric`` (``src/metrics/turn_taking_judge.py``) tests.

Pure units first (grid, rasteriser, backchannel proxy, upstream label state
machine, role swap), then the judge loop with an injected ``encode_fn``, then
the metric end to end with a fake VAD and an oracle judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    CHUNK,
    MIN_START,
    apply_backchannel_proxy,
    chunk_ends,
    label_rows,
    rasterise_channel,
    swap_roles,
)


# --------------------------------------------------------------------------- #
# Task 1: grid + rasteriser
# --------------------------------------------------------------------------- #
class TestGrid:
    def test_chunk_ends_match_upstream_grid(self):
        ends = chunk_ends(1.0)
        # int((1.0 - 0.2) / 0.04) = 20 chunks, ends 0.24 .. 1.00
        assert len(ends) == 20
        assert ends[0] == 0.24 and ends[-1] == 1.0
        assert all(isinstance(e, float) for e in ends)

    def test_chunk_ends_are_two_decimal_floats(self):
        ends = chunk_ends(24.0)
        assert ends == [
            float(f"{MIN_START + (i + 1) * CHUNK:.2f}") for i in range(len(ends))
        ]

    def test_rasterise_uses_upstream_inclusion_rule(self):
        ends = chunk_ends(1.0)  # 0.24 .. 1.00
        labels = rasterise_channel([(0.40, 0.60)], ends)
        # active iff s - 0.04 <= e < t: ends 0.36, 0.40, 0.44, 0.48, 0.52, 0.56
        active = [e for e, lab in zip(ends, labels) if lab == "IPU"]
        assert active == [0.36, 0.4, 0.44, 0.48, 0.52, 0.56]

    def test_rasterise_empty_is_all_silence(self):
        assert set(rasterise_channel([], chunk_ends(0.5))) == {"NA"}


# --------------------------------------------------------------------------- #
# Task 2: backchannel proxy
# --------------------------------------------------------------------------- #
class TestBackchannelProxy:
    @staticmethod
    def _kinds(out, ch):
        return [k for _, k in out[ch]]

    def test_short_span_inside_other_ipu_is_bc(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(2.0, 2.5)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]
        assert self._kinds(out, 0) == ["IPU"]

    def test_short_span_in_floor_holders_pause_is_bc(self):
        # ch0 holds the floor 0-2 and resumes 2.8-5; ch1's 2.1-2.5 is a BC.
        out = apply_backchannel_proxy([[(0.0, 2.0), (2.8, 5.0)], [(2.1, 2.5)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]

    def test_short_span_that_takes_the_floor_stays_ipu(self):
        # ch0 never speaks again before ch1's next IPU -> ch1 took the turn.
        out = apply_backchannel_proxy([[(0.0, 2.0)], [(2.1, 2.5), (3.0, 5.0)]], 6.0)
        assert self._kinds(out, 1) == ["IPU", "IPU"]

    def test_span_longer_than_cap_stays_ipu(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(1.0, 2.2)]], 6.0)
        assert self._kinds(out, 1) == ["IPU"]

    def test_cap_is_inclusive_at_1_08(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(1.0, 2.08)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]

    def test_first_utterance_of_the_window_is_never_bc(self):
        out = apply_backchannel_proxy([[(0.0, 0.5)], [(1.0, 5.0)]], 6.0)
        assert self._kinds(out, 0) == ["IPU"]

    def test_non_two_channel_input_is_all_ipu(self):
        out = apply_backchannel_proxy([[(0.0, 0.5)]], 6.0)
        assert self._kinds(out, 0) == ["IPU"]


# --------------------------------------------------------------------------- #
# Task 3: label rows (upstream state machine) + role swap
# --------------------------------------------------------------------------- #
def _labels(rows):
    return [r.split(",")[3] for r in rows]


def _turns(rows):
    return [r.split(",")[4] for r in rows]


class TestLabelRows:
    def test_row_format_and_grid(self):
        rows = label_rows("w", [[], []], 1.0)
        assert len(rows) == 20
        assert rows[0] == "w,0.2,0.24,NA,NA"
        assert rows[-1].startswith("w,0.96,1.0,")

    def test_turn_change_then_continuation(self):
        rows = label_rows("w", [[(0.2, 0.6)], [(0.8, 1.2)]], 1.4)
        labels = _labels(rows)
        first_ch0 = labels.index("T")
        assert labels[first_ch0 + 1] == "C"
        assert "NA" in labels[first_ch0 + 1 :]
        assert labels.count("T") == 2
        turns = _turns(rows)
        assert turns[0] == "NA" and "A" in turns and turns[-1] == "B"

    def test_pause_by_same_speaker_is_continuation_not_turn(self):
        rows = label_rows("w", [[(0.2, 0.6), (0.9, 1.3)], []], 1.4)
        assert _labels(rows).count("T") == 1

    def test_overlap_is_interruption_and_floor_passes(self):
        # ch0 0.2-1.0, ch1 0.6-1.8 (too long for the BC proxy): I during the
        # overlap, then ch1 alone -> T because the floor passes from AB to B.
        rows = label_rows("w", [[(0.2, 1.0)], [(0.6, 1.8)]], 2.0)
        labels = _labels(rows)
        assert "I" in labels
        i_last = len(labels) - 1 - labels[::-1].index("I")
        assert labels[i_last + 1] == "T"
        assert "AB" in _turns(rows)

    def test_interrupted_speaker_continuing_alone_is_c(self):
        rows = label_rows("w", [[(0.2, 2.0)], [(0.6, 1.8)]], 2.2)
        labels = _labels(rows)
        i_last = len(labels) - 1 - labels[::-1].index("I")
        assert labels[i_last + 1] == "C"

    def test_backchannel_inside_other_ipu(self):
        rows = label_rows("w", [[(0.2, 3.0)], [(1.0, 1.4)]], 3.2)
        labels = _labels(rows)
        assert "BC" in labels and "I" not in labels

    def test_bc_in_floor_holders_pause_is_plain_bc(self):
        # ch1's short span sits in ch0's pause and ch0 resumes: proxy -> BC,
        # state machine prev speaker A -> plain BC (never the BC_1 exception).
        rows = label_rows("w", [[(0.2, 1.0), (1.6, 3.0)], [(1.1, 1.4)]], 3.2)
        labels = _labels(rows)
        assert "BC" in labels and "BC_1" not in labels

    def test_single_channel_window_is_padded_with_silence(self):
        rows = label_rows("w", [[(0.2, 0.6)]], 1.0)
        assert _labels(rows).count("T") == 1 and "B" not in _turns(rows)

    def test_three_channels_raise(self):
        with pytest.raises(ValueError):
            label_rows("w", [[], [], []], 1.0)


class TestSwapRoles:
    def test_swaps_turn_column_only(self):
        rows = [
            "w,0.2,0.24,C,A",
            "w,0.24,0.28,I,AB",
            "w,0.28,0.32,NA,NA",
            "w,0.32,0.36,T,B",
        ]
        assert swap_roles(rows) == [
            "w,0.2,0.24,C,B",
            "w,0.24,0.28,I,BA",
            "w,0.28,0.32,NA,NA",
            "w,0.32,0.36,T,A",
        ]


# --------------------------------------------------------------------------- #
# Task 4: judge loop (run_chunk port) with an injected encoder
# --------------------------------------------------------------------------- #
from egs3.conversational.tts.src.metrics.turn_taking_judge import (  # noqa: E402
    TurnTakingJudge,
)


class TestJudgeLoop:
    def test_chunk_count_and_causal_window(self):
        seen = []

        def fake_encode(window):
            seen.append(len(window))
            return np.array([0.6, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        judge = TurnTakingJudge(encode_fn=fake_encode)
        wav = np.zeros(16000 * 35, dtype=np.float32)  # 35 s > 30 s context
        probs = judge.predict(wav)
        assert probs.shape == ((16000 * 35 - 3200) // 640, 5)
        assert seen[0] == 3200 + 640  # first window: start_chunk + one hop
        assert max(seen) == 480000  # never more than 30 s
        assert seen[-1] == 480000

    def test_chunk_count_matches_grid(self):
        judge = TurnTakingJudge(encode_fn=lambda w: np.full(5, 0.2, np.float32))
        wav = np.zeros(int(16000 * 7.37), dtype=np.float32)
        assert judge.predict(wav).shape[0] == len(chunk_ends(7.37))

    def test_short_audio_gives_zero_chunks(self):
        judge = TurnTakingJudge(encode_fn=lambda w: np.full(5, 0.2, np.float32))
        assert judge.predict(np.zeros(3000, np.float32)).shape == (0, 5)

    def test_likelihood_text_roundtrip(self):
        judge = TurnTakingJudge(
            encode_fn=lambda w: np.array([0.5, 0.2, 0.1, 0.1, 0.1], np.float32)
        )
        line = judge.likelihood_line(
            "w1", judge.predict(np.zeros(3200 + 640 * 2, np.float32))
        )
        assert line == "w1 0.5,0.2,0.1,0.1,0.1 0.5,0.2,0.1,0.1,0.1"


# --------------------------------------------------------------------------- #
# Tasks 5-6: the metric end to end (fake VAD + oracle judge), role pooling
# --------------------------------------------------------------------------- #
from egs3.conversational.tts.src.metrics import turn_taking_judge as m  # noqa: E402
from egs3.conversational.tts.src.metrics.turn_taking_judge import (  # noqa: E402
    ROLE_KEYS,
    TurnTakingJudgeMetric,
)

LAYER1_KEYS = {
    *(f"judge_f1_{c}" for c in ("C", "NA", "IN", "BC", "T")),
    "judge_f1_macro",
    *(f"judge_auc_{c}" for c in ("C", "NA", "IN", "BC", "T")),
    "judge_auc_mean",
    "judge_matched_chunks",
    "judge_expected_chunks",
    "judge_bc_proxy_count",
    "judge_skipped_windows",
}
ALL_KEYS = LAYER1_KEYS | set(ROLE_KEYS)


class KeyedFakeVADBackend:
    """Returns registered spans for the EXACT wav it's called with."""

    def __init__(self):
        self._table = {}

    @staticmethod
    def _key(wav):
        return tuple(np.round(np.asarray(wav, dtype=np.float64), 6).tolist())

    def register(self, wav, spans):
        arr = np.asarray(wav, dtype=np.float32)
        self._table[self._key(arr)] = list(spans)
        return arr

    def __call__(self, wav, sr):
        return self._table[self._key(wav)]


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _unique_wav(seed: int, n: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def _write_window(test_dir: Path, wid: str, duration_sec: float, gen_wavs, seed: int):
    sr = 16000
    channels = []
    for ch, gen in enumerate(gen_wavs):
        rel = f"wav/{wid}_ch{ch}.wav"
        _write_wav(test_dir / rel, gen, sr)
        channels.append(
            {"gen_wav": rel, "prompt_wav": rel, "gt_wav": rel, "ref_text": ""}
        )
    mix_rel = f"mix/{wid}.wav"
    _write_wav(test_dir / mix_rel, _unique_wav(seed, int(duration_sec * sr)), sr)
    meta = {
        "window_id": wid,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": sr,
        "num_channels": len(channels),
        "window_duration_sec": duration_sec,
        "mix_wav": mix_rel,
        "channels": channels,
        "turns": [],
    }
    (test_dir / "meta").mkdir(parents=True, exist_ok=True)
    (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_meta_scp(test_dir: Path, wids) -> None:
    (test_dir / "meta.scp").write_text("".join(f"{w} meta/{w}.json\n" for w in wids))


def _oracle_table(rows):
    """Judge likelihoods that put 0.9 on each chunk's realized class."""
    idx = {"C": 0, "NA": 1, "I": 2, "BC": 3, "T": 4}
    table = {}
    for r in rows:
        _, _, e, lab, _ = r.split(",")
        v = np.full(5, 0.025, np.float32)
        v[idx.get(lab, 1)] = 0.9
        table[float(e)] = v
    return table


class TestMetricLayer1:
    def test_oracle_judge_scores_perfect_and_keys(self, tmp_path):
        test_dir = tmp_path / "infer" / "valid"
        vad = KeyedFakeVADBackend()
        dur = 4.0
        g0 = vad.register(_unique_wav(1, 64000), [(0.2, 1.4)])
        g1 = vad.register(_unique_wav(2, 64000), [(1.8, 3.6)])
        _write_window(test_dir, "w1", dur, [g0, g1], seed=99)
        _write_meta_scp(test_dir, ["w1"])

        table = _oracle_table(label_rows("w1", [[(0.2, 1.4)], [(1.8, 3.6)]], dur))
        calls = {"n": 0}

        def encode(window):
            calls["n"] += 1
            return table[float(f"{(calls['n'] * 0.04 + 0.2):.2f}")]

        metric = TurnTakingJudgeMetric(
            judge=TurnTakingJudge(encode_fn=encode), vad_backend=vad
        )
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer")

        assert set(summary) == ALL_KEYS
        assert summary["judge_expected_chunks"] == 95
        assert summary["judge_matched_chunks"] == 95
        assert summary["judge_f1_C"] == 1.0
        assert summary["judge_f1_T"] == 1.0
        assert summary["judge_f1_NA"] == 1.0
        assert summary["judge_f1_IN"] is None and summary["judge_f1_BC"] is None
        assert summary["judge_f1_macro"] == 1.0
        assert summary["judge_auc_C"] == 1.0
        assert summary["judge_bc_proxy_count"] == 0
        assert all(summary[k] is None for k in ROLE_KEYS)  # flag off
        out = tmp_path / "infer" / "valid" / "scoring" / "turn_taking_judge"
        assert (out / "likelihoods" / "w1.txt").exists()
        assert (out / "labels" / "w1.txt").exists()
        assert (out / "confusion.json").exists()
        rec = json.loads((out / "windows.jsonl").read_text().splitlines()[0])
        assert rec["matched_chunks"] == 95 and rec["confusion"]["C"]["C"] > 0

    def test_cached_likelihoods_skip_the_judge(self, tmp_path):
        test_dir = tmp_path / "infer" / "valid"
        vad = KeyedFakeVADBackend()
        g0 = vad.register(_unique_wav(1, 32000), [(0.2, 1.0)])
        g1 = vad.register(_unique_wav(2, 32000), [])
        _write_window(test_dir, "w1", 2.0, [g0, g1], seed=7)
        _write_meta_scp(test_dir, ["w1"])
        calls = {"n": 0}

        def encode(window):
            calls["n"] += 1
            return np.array([0.6, 0.1, 0.1, 0.1, 0.1], np.float32)

        metric = TurnTakingJudgeMetric(
            judge=TurnTakingJudge(encode_fn=encode), vad_backend=vad
        )
        metric({"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer")
        first = calls["n"]
        assert first == 45
        metric({"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer")
        assert calls["n"] == first  # second pass read likelihoods/w1.txt

    def test_three_channel_window_is_skipped_not_scored(self, tmp_path):
        test_dir = tmp_path / "infer" / "valid"
        vad = KeyedFakeVADBackend()
        gens = [vad.register(_unique_wav(i, 32000), [(0.2, 1.0)]) for i in range(3)]
        _write_window(test_dir, "w1", 2.0, gens, seed=3)
        _write_meta_scp(test_dir, ["w1"])
        judge = TurnTakingJudge(encode_fn=lambda w: np.full(5, 0.2, np.float32))
        summary = TurnTakingJudgeMetric(judge=judge, vad_backend=vad)(
            {"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer"
        )
        assert summary["judge_skipped_windows"] == 1
        assert summary["judge_f1_macro"] is None

    def test_grid_drift_is_an_error(self, tmp_path):
        # A mixdown shorter than the meta duration yields fewer judge chunks
        # than label rows: the silent upstream skip must become an error.
        test_dir = tmp_path / "infer" / "valid"
        vad = KeyedFakeVADBackend()
        g0 = vad.register(_unique_wav(1, 64000), [(0.2, 1.0)])
        g1 = vad.register(_unique_wav(2, 64000), [])
        _write_window(test_dir, "w1", 4.0, [g0, g1], seed=5)
        _write_wav(test_dir / "mix" / "w1.wav", _unique_wav(6, 32000), 16000)
        _write_meta_scp(test_dir, ["w1"])
        judge = TurnTakingJudge(encode_fn=lambda w: np.full(5, 0.2, np.float32))
        with pytest.raises(RuntimeError, match="grid drift"):
            TurnTakingJudgeMetric(judge=judge, vad_backend=vad)(
                {"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer"
            )


class TestMetricLayer2:
    def test_role_metrics_pool_both_assignments(self):
        captured = []

        class FakeScore:
            def __init__(self, true_dict, pred_dict, turn_dict, labels, **kw):
                captured.append((kw, dict(turn_dict)))
                self.pred_arr = np.array(["C"])
                self.turn_arr = np.array(["B"])
                self.true_arr_soft_label = np.zeros((1, 5))
                self.true_arr_hard_label = np.array(["C"])

            def turn_change_metric(self):
                assert len(self.pred_arr) == 2  # pooled
                return (50.0, 100.0)

            def make_backchannel_metric(self):
                return (10.0, 20.0)

            def make_interruption_metric(self):
                return (30.0, 40.0)

            def turn_willingness_metric(self):
                return (60.0, 70.0)

            def handle_interruption_metric(self):
                return (80.0, None)

        rows = ["w,0.2,0.24,C,B"]
        metric = TurnTakingJudgeMetric(
            judge=TurnTakingJudge(encode_fn=lambda w: np.full(5, 0.2, np.float32)),
            report_role_metrics=True,
        )
        out = metric._role_metrics(FakeScore, {"w": {0.24: [0.2] * 5}}, rows)
        assert out["judge_acc_turn_change"] == 100.0
        assert out["judge_acc_interrupt_success"] is None
        assert set(out) == set(ROLE_KEYS)
        assert len(captured) == 2 and all(c[0].get("only_AI") for c in captured)
        assert captured[0][1]["w"][0.24] == "B" and captured[1][1]["w"][0.24] == "A"

    def test_role_metrics_run_on_real_upstream_library(self, tmp_path):
        # Upstream ScoreResult with a floor holder pausing and the other
        # speaker taking the turn: every key present, no exception.
        test_dir = tmp_path / "infer" / "valid"
        vad = KeyedFakeVADBackend()
        dur = 4.0
        g0 = vad.register(_unique_wav(1, 64000), [(0.2, 1.4)])
        g1 = vad.register(_unique_wav(2, 64000), [(1.8, 3.6)])
        _write_window(test_dir, "w1", dur, [g0, g1], seed=99)
        _write_meta_scp(test_dir, ["w1"])
        table = _oracle_table(label_rows("w1", [[(0.2, 1.4)], [(1.8, 3.6)]], dur))
        calls = {"n": 0}

        def encode(window):
            calls["n"] += 1
            return table[float(f"{(calls['n'] * 0.04 + 0.2):.2f}")]

        metric = TurnTakingJudgeMetric(
            judge=TurnTakingJudge(encode_fn=encode),
            vad_backend=vad,
            report_role_metrics=True,
        )
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", tmp_path / "infer")
        assert set(summary) == ALL_KEYS
        # ch1 took the turn after ch0's pause; oracle judge said T there.
        assert summary["judge_acc_turn_change"] == 100.0
        # undefined selections (no interruptions here) are None, never NaN
        assert summary["judge_acc_interrupt"] is None
        assert not any(isinstance(v, float) and np.isnan(v) for v in summary.values())
