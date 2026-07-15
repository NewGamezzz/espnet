"""``ConversationASRMetric`` (``src/metrics/asr.py``) tests.

Fake-transcriber / fake-VAD, CPU-only, no network: covers WER bookkeeping,
cpWER permutation + swap flag, pooled cross-channel script following (in
order, out of order, missing turn), the per-IPU timestamp offset, backend
laziness (transcriber/normalizer never import their real package before
first call), and a full ``__call__`` round trip against a fabricated
``inference_dir`` mirroring ``tests/test_measure.py``'s fixture pattern.

Real faster-whisper / openai-whisper are only exercised by the asset-gated
smoke test at the bottom (skipped when the packages are not installed,
following ``tests/test_pretrained_real.py``'s gating style).
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.asr import (
    ConversationASRMetric,
    FasterWhisperTranscriber,
    Word,
    WhisperEnglishNormalizer,
    _normalize_word,
)

try:
    import faster_whisper  # noqa: F401

    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False

try:
    import whisper.normalizers  # noqa: F401

    _HAS_OPENAI_WHISPER = True
except ImportError:
    _HAS_OPENAI_WHISPER = False


def _trivial_normalizer(text: str) -> str:
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------- #
# _cpwer: permutation search + identity tie-break
# --------------------------------------------------------------------------- #
class TestCpwer:
    def test_identity_is_optimal_no_swap(self):
        ref = ["hello world", "foo bar"]
        hyp = ["hello world", "foo bar"]
        wer, swap, perm = ConversationASRMetric._cpwer(ref, hyp)
        assert wer == pytest.approx(0.0)
        assert swap is False
        assert perm == (0, 1)

    def test_swapped_channels_are_detected(self):
        # channel 0's hypothesis is actually speaker 1's script, and vice
        # versa: the identity assignment is maximally wrong, the swap is
        # perfect.
        ref = ["hello world", "foo bar"]
        hyp = ["foo bar", "hello world"]
        wer, swap, perm = ConversationASRMetric._cpwer(ref, hyp)
        assert wer == pytest.approx(0.0)
        assert swap is True
        assert perm == (1, 0)

    def test_tie_prefers_identity(self):
        # Every permutation ties at wer=0 since both channels said the exact
        # same thing; the documented tie-break keeps the identity assignment.
        ref = ["a a", "a a"]
        hyp = ["a a", "a a"]
        _wer, swap, perm = ConversationASRMetric._cpwer(ref, hyp)
        assert swap is False
        assert perm == (0, 1)

    def test_partial_error_still_prefers_lower_wer_assignment(self):
        ref = ["one two", "four five"]
        hyp = ["one three", "four five"]  # 1 sub in channel 0 only
        wer, swap, perm = ConversationASRMetric._cpwer(ref, hyp)
        assert swap is False
        assert perm == (0, 1)
        assert wer == pytest.approx(0.25)  # 1 error / 4 total ref words

    def test_single_channel_is_trivially_identity(self):
        wer, swap, perm = ConversationASRMetric._cpwer(["hello"], ["hello"])
        assert wer == pytest.approx(0.0)
        assert swap is False
        assert perm == (0,)


# --------------------------------------------------------------------------- #
# pooled, cross-channel script following: _align_words_to_turns +
# _script_order_stats, exercised together the way _score_window combines them
# --------------------------------------------------------------------------- #
def _pooled_realized_time(per_channel_texts, per_channel_words, channel_of_turn):
    """Mirror _score_window's pooling: per-channel align, scatter into the
    window's chronological (``channel_of_turn``) order."""
    local = [
        ConversationASRMetric._align_words_to_turns(texts, words)
        for texts, words in zip(per_channel_texts, per_channel_words)
    ]
    cursor = [0] * len(per_channel_texts)
    pooled = []
    for ch in channel_of_turn:
        idx = cursor[ch]
        pooled.append(local[ch][idx])
        cursor[ch] += 1
    return pooled


class TestScriptFollowingPooling:
    def test_in_order_cross_channel_turns_score_perfectly(self):
        # Global script order: ch0 "hello world" (gi0), ch1 "foo bar" (gi1),
        # ch0 "baz qux" (gi2). Word times respect that order.
        ch0_texts = ["hello world", "baz qux"]
        ch0_words = [
            Word("hello", 0.0, 0.0),
            Word("world", 0.5, 0.5),
            Word("baz", 4.0, 4.0),
            Word("qux", 4.5, 4.5),
        ]
        ch1_texts = ["foo bar"]
        ch1_words = [Word("foo", 2.0, 2.0), Word("bar", 2.5, 2.5)]

        pooled = _pooled_realized_time(
            [ch0_texts, ch1_texts], [ch0_words, ch1_words], channel_of_turn=[0, 1, 0]
        )
        assert pooled == [pytest.approx(0.25), pytest.approx(2.25), pytest.approx(4.25)]

        stats = ConversationASRMetric._script_order_stats(pooled)
        assert stats["turn_order_acc"] == pytest.approx(1.0)
        assert stats["kendall_tau"] == pytest.approx(1.0)
        assert stats["turn_count_ratio"] == pytest.approx(1.0)
        assert stats["missing_turns"] == []

    def test_out_of_order_cross_channel_turn_is_penalized(self):
        # Same script as above, but channel 1 says its turn TOO EARLY (before
        # channel 0's first scripted turn) -- a real cross-speaker ordering
        # violation that a per-channel-only metric could never see (channel
        # 1 only has one turn, so it can't be "out of order" with itself).
        ch0_texts = ["hello world", "baz qux"]
        ch0_words = [
            Word("hello", 0.0, 0.0),
            Word("world", 0.5, 0.5),
            Word("baz", 4.0, 4.0),
            Word("qux", 4.5, 4.5),
        ]
        ch1_texts = ["foo bar"]
        ch1_words = [Word("foo", -1.0, -1.0), Word("bar", -0.5, -0.5)]

        pooled = _pooled_realized_time(
            [ch0_texts, ch1_texts], [ch0_words, ch1_words], channel_of_turn=[0, 1, 0]
        )
        assert pooled == [
            pytest.approx(0.25),
            pytest.approx(-0.75),
            pytest.approx(4.25),
        ]

        stats = ConversationASRMetric._script_order_stats(pooled)
        # script order [0,1,2]; time order is [1,0,2] (gi1 said earliest) ->
        # only gi2's rank (2nd->2nd) matches -> 1/3.
        assert stats["turn_order_acc"] == pytest.approx(1 / 3)
        # 2 concordant pairs, (0,2) and (1,2); 1 discordant pair (0,1) ->
        # tau = (2 - 1) / 3.
        assert stats["kendall_tau"] == pytest.approx(1 / 3)
        assert stats["turn_count_ratio"] == pytest.approx(1.0)

    def test_missing_turn_excluded_from_order_and_tau_but_counted_in_ratio(self):
        # channel 0 never says its second scripted turn ("baz qux") at all.
        ch0_texts = ["hello world", "baz qux"]
        ch0_words = [Word("hello", 0.0, 0.0), Word("world", 0.5, 0.5)]
        ch1_texts = ["foo bar"]
        ch1_words = [Word("foo", 2.0, 2.0), Word("bar", 2.5, 2.5)]

        pooled = _pooled_realized_time(
            [ch0_texts, ch1_texts], [ch0_words, ch1_words], channel_of_turn=[0, 1, 0]
        )
        assert pooled == [pytest.approx(0.25), pytest.approx(2.25), None]

        stats = ConversationASRMetric._script_order_stats(pooled)
        assert stats["missing_turns"] == [2]
        assert stats["num_realized_turns"] == 2
        assert stats["num_scripted_turns"] == 3
        assert stats["turn_count_ratio"] == pytest.approx(2 / 3)
        # The two realized turns (gi0, gi1) are still in correct relative
        # order, so accuracy/tau over just that pair are perfect.
        assert stats["turn_order_acc"] == pytest.approx(1.0)
        assert stats["kendall_tau"] == pytest.approx(1.0)

    def test_no_scripted_turns_returns_all_none(self):
        stats = ConversationASRMetric._script_order_stats([])
        assert stats["num_scripted_turns"] == 0
        assert stats["turn_order_acc"] is None
        assert stats["kendall_tau"] is None
        assert stats["turn_count_ratio"] is None

    def test_single_realized_turn_has_no_kendall_tau(self):
        stats = ConversationASRMetric._script_order_stats([0.5])
        assert stats["turn_order_acc"] == pytest.approx(1.0)
        assert stats["kendall_tau"] is None
        assert stats["turn_count_ratio"] == pytest.approx(1.0)


class TestNormalizeWord:
    def test_lowercases_and_strips_punctuation(self):
        assert _normalize_word("Hello,") == "hello"
        assert _normalize_word("--World!!") == "world"

    def test_keeps_internal_apostrophes(self):
        assert _normalize_word("don't") == "don't"

    def test_empty_after_stripping_punctuation_only(self):
        assert _normalize_word("...") == ""


# --------------------------------------------------------------------------- #
# _transcribe_channel: per-IPU transcription + timestamp offset
# --------------------------------------------------------------------------- #
class _ListVAD:
    def __init__(self, segments):
        self._segments = segments

    def __call__(self, wav, sr):
        return self._segments


class _QueueTranscriber:
    """Pops one canned word-list per call, in call order."""

    def __init__(self, calls):
        self._queue = list(calls)

    def __call__(self, wav, sr):
        return self._queue.pop(0)


class TestTranscribeChannelOffsetsByIpuStart:
    def test_second_ipu_words_are_offset_by_its_own_start(self, tmp_path):
        sr = 16000
        wav = np.zeros(int(sr * 3.0), dtype=np.float32)
        path = tmp_path / "ch.wav"
        sf.write(str(path), wav, sr)

        # Two well-separated IPUs (gap 1.5s >> default min_silence=0.2s), no
        # padding so the reported IPU bounds are exactly these.
        vad = _ListVAD([(0.0, 0.5), (2.0, 2.5)])
        transcriber = _QueueTranscriber(
            [
                [Word("hi", 0.0, 0.1)],  # 1st IPU call: local time
                [Word("there", 0.05, 0.15)],  # 2nd IPU call: local time
            ]
        )
        metric = ConversationASRMetric(
            transcriber=transcriber,
            normalizer=_trivial_normalizer,
            vad=vad,
            pad=0.0,
        )

        words = metric._transcribe_channel(path)

        assert [w.text for w in words] == ["hi", "there"]
        assert words[0].start == pytest.approx(0.0)
        assert words[0].end == pytest.approx(0.1)
        # Offset by the 2nd IPU's OWN start (2.0), not the 1st IPU's.
        assert words[1].start == pytest.approx(2.05)
        assert words[1].end == pytest.approx(2.15)

    def test_no_ipus_yields_no_words_and_transcriber_is_never_called(self, tmp_path):
        sr = 16000
        wav = np.zeros(sr, dtype=np.float32)
        path = tmp_path / "ch.wav"
        sf.write(str(path), wav, sr)

        def _boom(_wav, _sr):
            raise AssertionError("transcriber must not be called with zero IPUs")

        metric = ConversationASRMetric(
            transcriber=_boom, normalizer=_trivial_normalizer, vad=_ListVAD([])
        )
        assert metric._transcribe_channel(path) == []


# --------------------------------------------------------------------------- #
# backend laziness: constructing the real defaults must never import their
# heavy package; only the first call may.
# --------------------------------------------------------------------------- #
class TestBackendLaziness:
    def test_faster_whisper_transcriber_construction_does_not_import_it(
        self, monkeypatch
    ):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "faster_whisper" or name.startswith("faster_whisper."):
                raise AssertionError("faster_whisper imported before first call")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        transcriber = FasterWhisperTranscriber()
        assert transcriber._model is None

    def test_whisper_normalizer_construction_does_not_import_it(self, monkeypatch):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "whisper" or name.startswith("whisper."):
                raise AssertionError("whisper imported before first call")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        normalizer = WhisperEnglishNormalizer()
        assert normalizer._normalizer is None

    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        # Constructing ConversationASRMetric() with no injected backends must
        # succeed offline: every real default is lazy.
        metric = ConversationASRMetric()
        assert isinstance(metric.transcriber, FasterWhisperTranscriber)
        assert isinstance(metric.normalizer, WhisperEnglishNormalizer)

    @pytest.mark.skipif(
        _HAS_OPENAI_WHISPER, reason="openai-whisper is installed in this environment"
    )
    def test_default_normalizer_raises_with_install_hint_when_missing(self):
        normalizer = WhisperEnglishNormalizer()
        with pytest.raises(ImportError, match="openai-whisper"):
            normalizer("hello world")


# --------------------------------------------------------------------------- #
# full __call__ round trip against a fabricated inference_dir, mirroring
# tests/test_measure.py's fixture pattern (same meta-JSON shape as
# src/inference.py's real output contract).
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(duration_s * sr), dtype=np.float32), sr)


def _write_window(
    test_dir: Path,
    window_id: str,
    ch0_ref: str,
    ch1_ref: str,
    boundary: float = 5.0,
) -> None:
    for ch in (0, 1):
        _write_wav(test_dir / "wav" / f"{window_id}_ch{ch}.wav")
    meta = {
        "window_id": window_id,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": 24000,
        "num_channels": 2,
        "prompt_boundary_sec": boundary,
        "prompt_boundary_frames": 469,
        "window_duration_sec": 12.0,
        "rtf": 0.5,
        "channels": [
            {
                "gen_wav": f"wav/{window_id}_ch0.wav",
                "prompt_wav": f"prompt/{window_id}_ch0.wav",
                "gt_wav": f"gt/{window_id}_ch0.wav",
                "ref_text": ch0_ref,
            },
            {
                "gen_wav": f"wav/{window_id}_ch1.wav",
                "prompt_wav": f"prompt/{window_id}_ch1.wav",
                "gt_wav": f"gt/{window_id}_ch1.wav",
                "ref_text": ch1_ref,
            },
        ],
        "turns": [
            {"channel": 0, "text": ch0_ref, "start": boundary, "end": boundary + 1.0},
            {
                "channel": 1,
                "text": ch1_ref,
                "start": boundary + 1.5,
                "end": boundary + 2.5,
            },
        ],
    }
    (test_dir / "meta").mkdir(parents=True, exist_ok=True)
    (test_dir / f"meta/{window_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_meta_scp(test_dir: Path, window_ids: list[str]) -> None:
    lines = [f"{wid} meta/{wid}.json" for wid in window_ids]
    (test_dir / "meta.scp").write_text("".join(f"{line}\n" for line in lines))


class TestCallRoundTrip:
    def _build_inference_dir(self, tmp_path) -> Path:
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        # w1: both channels transcribed perfectly, ch1's turn later than
        # ch0's -> matches script order.
        _write_window(test_dir, "sess_w00000", "alpha beta", "gamma delta")
        # w2: channel 0 has one substitution error; channel 1 still exact.
        _write_window(test_dir, "sess_w00001", "one two", "four five")
        _write_meta_scp(test_dir, ["sess_w00000", "sess_w00001"])
        return inference_dir

    def _metric(self) -> ConversationASRMetric:
        # One call per channel per window (whole-file IPU); order follows
        # meta.scp order (w1 then w2) x channel order (ch0 then ch1).
        transcriber = _QueueTranscriber(
            [
                [Word("alpha", 0.0, 0.0), Word("beta", 0.1, 0.1)],  # w1 ch0 (exact)
                [Word("gamma", 1.0, 1.0), Word("delta", 1.1, 1.1)],  # w1 ch1 (exact)
                [Word("one", 0.0, 0.0), Word("three", 0.1, 0.1)],  # w2 ch0 (1 sub)
                [Word("four", 1.0, 1.0), Word("five", 1.1, 1.1)],  # w2 ch1 (exact)
            ]
        )
        return ConversationASRMetric(
            transcriber=transcriber,
            normalizer=_trivial_normalizer,
            vad=_ListVAD([(0.0, 0.5)]),
            pad=0.0,
        )

    def test_summary_matches_hand_derived_values(self, tmp_path):
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)

        # window 1: wer=[0,0] -> mean/worst 0; cpwer 0; in-order turns.
        # window 2: wer=[0.5,0] -> mean 0.25, worst 0.5; cpwer 0.25 (identity
        # beats the swapped assignment); still in-order turns.
        assert summary["wer_ch_mean"] == pytest.approx((0.0 + 0.25) / 2)
        assert summary["wer_ch_worst"] == pytest.approx((0.0 + 0.5) / 2)
        assert summary["cpwer"] == pytest.approx((0.0 + 0.25) / 2)
        assert summary["swap_rate"] == pytest.approx(0.0)
        assert summary["turn_order_acc"] == pytest.approx(1.0)
        assert summary["kendall_tau"] == pytest.approx(1.0)
        assert summary["turn_count_ratio"] == pytest.approx(1.0)
        assert set(summary) == {
            "wer_ch_mean",
            "wer_ch_worst",
            "cpwer",
            "swap_rate",
            "turn_order_acc",
            "kendall_tau",
            "turn_count_ratio",
        }

    def test_writes_jsonl_and_summary_artifacts_under_scoring_dir(self, tmp_path):
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "conversation_asr"
        jsonl_lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(jsonl_lines) == 2
        records = [json.loads(line) for line in jsonl_lines]
        assert [r["window_id"] for r in records] == ["sess_w00000", "sess_w00001"]

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        # Same invariant tests/test_measure.py checks for the stub metric:
        # gen_wav paths in the meta JSON are relative to inference_dir/valid,
        # and this metric must actually open the files there (not merely
        # parse JSON structurally). A wrong test_dir would raise FileNotFound
        # when soundfile tries to read the (nonexistent) resolved path.
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)
        assert summary["wer_ch_mean"] == pytest.approx(0.125)


class TestCallRoundTripOutOfOrderDetection:
    """The headline capability -- detecting a CROSS-CHANNEL turn-order
    violation -- exercised through the real ``__call__``/``_score_window``
    scatter, not the ``_pooled_realized_time`` test helper. This is the
    same pooling code path production runs, so a scatter bug (e.g. an
    off-by-one in ``channel_positions``) would show up here even though the
    pure-function tests in ``TestScriptFollowingPooling`` already prove the
    aggregation math itself is correct.
    """

    def test_channel_speaking_out_of_script_order_lowers_turn_order_acc(
        self, tmp_path
    ):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        # Script order (per _write_window): ch0's turn @ boundary=5.0, then
        # ch1's turn @ 6.5 -- ch1 is scripted to speak SECOND.
        _write_window(test_dir, "sess_w00000", "hello world", "foo bar")
        _write_meta_scp(test_dir, ["sess_w00000"])

        # But channel 1 actually speaks FIRST (negative local time, well
        # before channel 0's realized time) -- a real interleaving
        # violation between speakers, exactly the failure mode PLAN-step4.md
        # calls out ("script order violated").
        transcriber = _QueueTranscriber(
            [
                [Word("hello", 5.0, 5.0), Word("world", 5.5, 5.5)],  # ch0
                [Word("foo", -1.0, -1.0), Word("bar", -0.5, -0.5)],  # ch1
            ]
        )
        metric = ConversationASRMetric(
            transcriber=transcriber,
            normalizer=_trivial_normalizer,
            vad=_ListVAD([(0.0, 0.5)]),
            pad=0.0,
        )
        data = {"meta": test_dir / "meta.scp"}

        summary = metric(data, "valid", inference_dir)

        # 2 realized turns, actual time order reversed vs script order ->
        # 0/2 positions match, and the single pair is discordant.
        assert summary["turn_order_acc"] == pytest.approx(0.0)
        assert summary["kendall_tau"] == pytest.approx(-1.0)
        assert summary["turn_count_ratio"] == pytest.approx(1.0)
        # WER/cpWER are unaffected by ordering -- both channels still
        # transcribed exactly, sanity-checking the two behaviors are
        # independent in the shipped code.
        assert summary["wer_ch_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# asset-gated real-backend smoke: skipped unless faster-whisper AND
# openai-whisper are actually installed (neither is available locally per
# the task brief; this closes the gap on a machine that has them).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_HAS_FASTER_WHISPER and _HAS_OPENAI_WHISPER),
    reason="faster-whisper and/or openai-whisper not installed",
)
class TestRealBackendsSmoke:
    def test_real_normalizer_lowercases_and_expands(self):
        normalizer = WhisperEnglishNormalizer()
        out = normalizer("Twenty-Five dollars, please!")
        assert out == out.lower()
        assert out.strip() != ""

    def test_real_transcriber_does_not_crash_on_silence(self):
        transcriber = FasterWhisperTranscriber(model_size="tiny")
        silence = np.zeros(16000, dtype=np.float32)
        words = transcriber(silence, 16000)
        assert isinstance(words, list)
