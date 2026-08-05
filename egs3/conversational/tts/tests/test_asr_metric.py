"""``ConversationASRMetric`` (``src/metrics/asr.py``) tests.

Fake-transcriber, CPU-only, no network: covers the corpus-level (pooled)
counting primitives (proving pooling is genuinely NOT a mean of
per-utterance WERs), the mixed-channel reference's start-time ordering, the
normalizer applying to both hypothesis and reference, backend laziness
(transcriber/normalizer never import their real package before first call),
and a full ``__call__`` round trip against a fabricated ``inference_dir``
matching ``src/inference.py``'s current meta contract (top-level ``mix_wav``,
no ``prompt_boundary_sec``/``prompt_boundary_frames``).

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
    WhisperEnglishNormalizer,
    _counts,
    _mix_reference,
    _pool_wer,
    _wer_from_counts,
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
# corpus-level (pooled) WER counting: the core "never average per-utterance
# WERs" primitive, tested as pure functions.
# --------------------------------------------------------------------------- #
class TestWerFromCounts:
    def test_exact_match_is_zero(self):
        assert _wer_from_counts(_counts("a b", "a b")) == pytest.approx(0.0)

    def test_one_substitution(self):
        # ref "a b", hyp "a c": 1 sub / 2 ref words.
        assert _wer_from_counts(_counts("a b", "a c")) == pytest.approx(0.5)

    def test_zero_reference_words_is_none_not_a_fabricated_number(self):
        # An insertion-only utterance (empty reference) has an undefined
        # denominator (S+D+H == 0); this must be None, not a fabricated 0.0
        # or a ZeroDivisionError.
        assert _wer_from_counts(_counts("", "a b")) is None


class TestPoolWer:
    def test_pooled_wer_differs_from_mean_of_per_utterance_wer(self):
        # Utterance A: 2 ref words, 1 substitution -> wer 0.5.
        # Utterance B: 8 ref words, 1 substitution -> wer 0.125.
        # A NAIVE mean-of-WERs would give (0.5 + 0.125) / 2 = 0.3125; pooling
        # the raw counts instead (2 subs / 10 total ref words) gives 0.2 --
        # the corpus-WER convention this metric is required to use.
        counts_a = _counts("a b", "a c")
        counts_b = _counts("a b c d e f g h", "a b c d e f g x")
        wer_a = _wer_from_counts(counts_a)
        wer_b = _wer_from_counts(counts_b)
        mean_of_wers = (wer_a + wer_b) / 2.0

        pooled = _pool_wer([counts_a, counts_b])

        assert pooled == pytest.approx(0.2)
        assert pooled != pytest.approx(mean_of_wers)

    def test_empty_pool_is_none(self):
        assert _pool_wer([]) is None

    def test_single_utterance_pool_equals_its_own_wer(self):
        counts = _counts("a b", "a c")
        assert _pool_wer([counts]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# _mix_reference: start-time ordering across channels
# --------------------------------------------------------------------------- #
class TestMixReference:
    def test_orders_turns_by_start_time_regardless_of_list_order(self):
        turns = [
            {"channel": 1, "text": "second", "start": 5.0, "end": 6.0},
            {"channel": 0, "text": "first", "start": 1.0, "end": 2.0},
        ]
        assert _mix_reference(turns) == "first second"

    def test_ties_keep_the_existing_list_order(self):
        turns = [
            {"channel": 0, "text": "a", "start": 1.0, "end": 2.0},
            {"channel": 1, "text": "b", "start": 1.0, "end": 2.0},
        ]
        assert _mix_reference(turns) == "a b"
        # Same tie, reversed list order -> the join flips too, proving the
        # sort is stable (keys the tie on original position) rather than
        # imposing some other deterministic order (e.g. channel index).
        assert _mix_reference(list(reversed(turns))) == "b a"

    def test_empty_turns_is_empty_string(self):
        assert _mix_reference([]) == ""


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
# decode settings actually passed to faster-whisper: long channel files
# derail whisper's conditioned long-form decoding after silent/garbled spans
# (chunked CoVoMix2 dialogues lost whole spoken final chunks to it, debugged
# 2026-08-05), so conditioning on previous text must be OFF by default and
# the whole kwarg set is pinned here as a contract.
# --------------------------------------------------------------------------- #
class _RecordingModel:
    """Stands in for faster_whisper.WhisperModel: records transcribe kwargs."""

    def __init__(self):
        self.calls = []

    def transcribe(self, wav, **kwargs):
        self.calls.append(kwargs)
        return iter(()), None


class TestTranscriberDecodeSettings:
    def _call(self, transcriber):
        model = _RecordingModel()
        transcriber._model = model  # pre-loaded: _load() becomes a no-op
        transcriber(np.zeros(16000, dtype=np.float32), 16000)
        assert len(model.calls) == 1
        return model.calls[0]

    def test_conditioning_on_previous_text_is_off_by_default(self):
        kwargs = self._call(FasterWhisperTranscriber())
        assert kwargs["condition_on_previous_text"] is False

    def test_conditioning_can_be_reenabled_to_reproduce_old_scoring(self):
        kwargs = self._call(FasterWhisperTranscriber(condition_on_previous_text=True))
        assert kwargs["condition_on_previous_text"] is True

    def test_vad_filter_and_language_stay_pinned(self):
        kwargs = self._call(FasterWhisperTranscriber())
        assert kwargs["vad_filter"] is True
        assert kwargs["language"] == "en"


# --------------------------------------------------------------------------- #
# full __call__ round trip against a fabricated inference_dir, matching
# src/inference.py's current meta contract (module docstring): top-level
# mix_wav, channels[ch].{gen_wav,prompt_wav,gt_wav,ref_text}, turns.
# --------------------------------------------------------------------------- #
class _QueueTranscriber:
    """Pops one canned hypothesis string per call, in call order."""

    def __init__(self, calls):
        self._queue = list(calls)

    def __call__(self, wav, sr):
        return self._queue.pop(0)


def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(duration_s * sr), dtype=np.float32), sr)


def _write_window(
    test_dir: Path,
    window_id: str,
    *refs: str,
    turn_starts: list[float] | None = None,
) -> None:
    """One fabricated window in the CURRENT infer-stage meta contract, one
    channel per positional ``refs`` entry. ``turn_starts`` (default:
    ascending, one per channel) lets a test control cross-channel start-time
    ordering for ``wer_mix``."""
    n = len(refs)
    if turn_starts is None:
        turn_starts = [1.5 * ch for ch in range(n)]
    for ch in range(n):
        _write_wav(test_dir / "wav" / f"{window_id}_ch{ch}.wav")
        _write_wav(test_dir / "prompt" / f"{window_id}_ch{ch}.wav")
        _write_wav(test_dir / "gt" / f"{window_id}_ch{ch}.wav")
    _write_wav(test_dir / "mix" / f"{window_id}.wav")
    meta = {
        "window_id": window_id,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": 24000,
        "num_channels": n,
        "window_duration_sec": 12.0,
        "rtf": 0.5,
        "mix_wav": f"mix/{window_id}.wav",
        "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
        "channels": [
            {
                "gen_wav": f"wav/{window_id}_ch{ch}.wav",
                "prompt_wav": f"prompt/{window_id}_ch{ch}.wav",
                "gt_wav": f"gt/{window_id}_ch{ch}.wav",
                "ref_text": ref,
            }
            for ch, ref in enumerate(refs)
        ],
        "turns": [
            {
                "channel": ch,
                "text": ref,
                "start": turn_starts[ch],
                "end": turn_starts[ch] + 1.0,
            }
            for ch, ref in enumerate(refs)
        ],
    }
    (test_dir / "meta").mkdir(parents=True, exist_ok=True)
    (test_dir / f"meta/{window_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_meta_scp(test_dir: Path, window_ids: "list[str]") -> None:
    lines = [f"{wid} meta/{wid}.json" for wid in window_ids]
    (test_dir / "meta.scp").write_text("".join(f"{line}\n" for line in lines))


class TestCallRoundTrip:
    def _build_inference_dir(self, tmp_path) -> Path:
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        # w1: both channels transcribed perfectly.
        _write_window(test_dir, "sess_w00000", "alpha beta", "gamma delta")
        # w2: channel 0 has one substitution error; channel 1 is exact.
        _write_window(test_dir, "sess_w00001", "one two", "four five")
        _write_meta_scp(test_dir, ["sess_w00000", "sess_w00001"])
        return inference_dir

    def _metric(self) -> ConversationASRMetric:
        # One call per channel + one per mix, per window, in meta.scp order.
        transcriber = _QueueTranscriber(
            [
                "alpha beta",  # w1 ch0 (exact)
                "gamma delta",  # w1 ch1 (exact)
                "alpha beta gamma delta",  # w1 mix (exact)
                "one three",  # w2 ch0 (1 sub: two->three)
                "four five",  # w2 ch1 (exact)
                "one three four five",  # w2 mix (1 sub)
            ]
        )
        return ConversationASRMetric(
            transcriber=transcriber, normalizer=_trivial_normalizer
        )

    def test_summary_matches_hand_pooled_counts(self, tmp_path):
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)

        # wer_channel: 4 channel utterances, refs = [2,2,2,2] words, one sub
        # total (w2 ch0) -> pooled (0+0+1)/(0+0+2+2+2+2-... ) let's just
        # pool directly: S=1,D=0,I=0,H=(2+2+1+2)=7 -> wer = 1/8 = 0.125.
        assert summary["wer_channel"] == pytest.approx(0.125)
        # wer_mix: 2 window-utterances, refs = [4,4] words, one sub total
        # (w2 mix) -> S=1,D=0,I=0,H=(4+3)=7 -> wer = 1/8 = 0.125.
        assert summary["wer_mix"] == pytest.approx(0.125)
        assert set(summary) == {"wer_channel", "wer_mix"}

    def test_writes_jsonl_and_summary_artifacts_under_scoring_dir(self, tmp_path):
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "conversation_asr"
        jsonl_lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(jsonl_lines) == 2
        records = [json.loads(line) for line in jsonl_lines]
        assert [r["window_id"] for r in records] == ["sess_w00000", "sess_w00001"]
        # Debug-only per-window WER is present but is NOT what the summary
        # is computed from (that's the pooled-counts path, tested above).
        assert records[1]["channels"][0]["wer"] == pytest.approx(0.5)

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        # Same invariant tests/test_measure.py checks for the stub metric:
        # gen_wav/mix_wav paths in the meta JSON are relative to
        # inference_dir/valid, and this metric must actually open the files
        # there (not merely parse JSON structurally).
        inference_dir = self._build_inference_dir(tmp_path)
        data = {"meta": inference_dir / "valid" / "meta.scp"}

        summary = self._metric()(data, "valid", inference_dir)
        assert summary["wer_channel"] == pytest.approx(0.125)


class TestCallRoundTripMixOrdering:
    def test_wer_mix_reference_follows_start_time_not_channel_index(self, tmp_path):
        # Channel 1's turn (per turn_starts) happens BEFORE channel 0's, so
        # the mix reference must read "beta alpha" (time order), not
        # "alpha beta" (channel-index order) -- exercised through the real
        # __call__ / _score_window path, not the pure _mix_reference helper.
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_window(test_dir, "sess_w00000", "alpha", "beta", turn_starts=[5.0, 1.0])
        _write_meta_scp(test_dir, ["sess_w00000"])

        transcriber = _QueueTranscriber(
            ["alpha", "beta", "beta alpha"]  # ch0, ch1, mix (perfect mix hyp)
        )
        metric = ConversationASRMetric(
            transcriber=transcriber, normalizer=_trivial_normalizer
        )
        data = {"meta": test_dir / "meta.scp"}

        summary = metric(data, "valid", inference_dir)

        assert summary["wer_mix"] == pytest.approx(0.0)

        scoring_dir = inference_dir / "valid" / "scoring" / "conversation_asr"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["mix"]["ref_text"] == "beta alpha"


class TestCallRoundTripNormalization:
    def test_normalizer_applies_to_both_hypothesis_and_reference(self, tmp_path):
        # ref_text is upper-case, the fake transcriber's hypothesis is
        # lower-case; without normalizing BOTH sides to the same case, jiwer
        # (case-sensitive) would score every word as an error. A lowering
        # normalizer applied symmetrically must make this a perfect match.
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_window(test_dir, "sess_w00000", "HELLO WORLD")
        _write_meta_scp(test_dir, ["sess_w00000"])

        transcriber = _QueueTranscriber(["hello world", "hello world"])
        metric = ConversationASRMetric(transcriber=transcriber, normalizer=str.lower)
        data = {"meta": test_dir / "meta.scp"}

        summary = metric(data, "valid", inference_dir)

        assert summary["wer_channel"] == pytest.approx(0.0)
        assert summary["wer_mix"] == pytest.approx(0.0)


class TestCallRoundTripEdgeCases:
    def test_single_channel_window_does_not_crash(self, tmp_path):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_window(test_dir, "sess_w00000", "hello world")
        _write_meta_scp(test_dir, ["sess_w00000"])

        transcriber = _QueueTranscriber(["hello world", "hello world"])
        metric = ConversationASRMetric(
            transcriber=transcriber, normalizer=_trivial_normalizer
        )
        data = {"meta": test_dir / "meta.scp"}

        summary = metric(data, "valid", inference_dir)

        assert summary["wer_channel"] == pytest.approx(0.0)
        assert summary["wer_mix"] == pytest.approx(0.0)

    def test_empty_reference_text_channel_does_not_crash(self, tmp_path):
        # ch0 has no scripted reference text at all (e.g. an entirely-silent
        # generated channel); ch1 is normal. The window-level turns list
        # only has ch1's turn (ch0 contributed no turn), so wer_mix's
        # reference is unaffected.
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        test_dir.mkdir(parents=True)
        _write_wav(test_dir / "wav" / "sess_w00000_ch0.wav")
        _write_wav(test_dir / "wav" / "sess_w00000_ch1.wav")
        _write_wav(test_dir / "prompt" / "sess_w00000_ch0.wav")
        _write_wav(test_dir / "prompt" / "sess_w00000_ch1.wav")
        _write_wav(test_dir / "mix" / "sess_w00000.wav")
        meta = {
            "window_id": "sess_w00000",
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": 24000,
            "num_channels": 2,
            "window_duration_sec": 12.0,
            "rtf": 0.5,
            "mix_wav": "mix/sess_w00000.wav",
            "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
            "channels": [
                {
                    "gen_wav": "wav/sess_w00000_ch0.wav",
                    "prompt_wav": "prompt/sess_w00000_ch0.wav",
                    "gt_wav": "wav/sess_w00000_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": "wav/sess_w00000_ch1.wav",
                    "prompt_wav": "prompt/sess_w00000_ch1.wav",
                    "gt_wav": "wav/sess_w00000_ch1.wav",
                    "ref_text": "gamma delta",
                },
            ],
            "turns": [{"channel": 1, "text": "gamma delta", "start": 1.5, "end": 2.5}],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / "meta/sess_w00000.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        _write_meta_scp(test_dir, ["sess_w00000"])

        transcriber = _QueueTranscriber(
            ["", "gamma delta", "gamma delta"]  # ch0, ch1, mix
        )
        metric = ConversationASRMetric(
            transcriber=transcriber, normalizer=_trivial_normalizer
        )
        data = {"meta": test_dir / "meta.scp"}

        summary = metric(data, "valid", inference_dir)

        # ch0: ref="" hyp="" -> S=D=I=0, H=0 -> per-utterance WER undefined,
        # but the pooled denominator still has ch1's 2 ref words -> defined.
        assert summary["wer_channel"] == pytest.approx(0.0)
        assert summary["wer_mix"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: the binding constraint that the shipped config
# instantiates every metric offline with its real (lazy) defaults, i.e.
# constructing ConversationASRMetric() from the config never downloads.
# Mirrors the equivalent test in test_speaker_metric.py / test_quality_metric.py.
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_conversation_asr_metric_entry_instantiates_without_network(
        self, monkeypatch
    ):
        from hydra.utils import instantiate

        from egs3.conversational.tts import run
        from espnet3.utils.config_utils import load_and_merge_config

        recipe_dir = Path(run.__file__).resolve().parent
        monkeypatch.chdir(recipe_dir)
        metrics_config = load_and_merge_config(
            Path("conf/metrics.yaml"),
            config_name=run.DEFAULT_METRICS_CONFIG,
            resolve=False,
        )

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name in ("faster_whisper", "whisper") or name.startswith(
                ("faster_whisper.", "whisper.")
            ):
                raise AssertionError(f"{name} imported while instantiating config")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)

        entries = [
            entry
            for entry in metrics_config.metrics
            if entry.metric._target_.endswith("ConversationASRMetric")
        ]
        assert len(entries) == 1
        metric = instantiate(entries[0].metric)
        assert isinstance(metric, ConversationASRMetric)
        assert isinstance(metric.transcriber, FasterWhisperTranscriber)
        assert metric.transcriber._model is None
        assert isinstance(metric.normalizer, WhisperEnglishNormalizer)
        assert metric.normalizer._normalizer is None

    # The all-three-entries-instantiate-together check already lives in
    # tests/test_speaker_metric.py's TestMetricsConfigInstantiatesOffline
    # (test_every_configured_metric_instantiates_without_network); no need
    # to duplicate that loop here.


# --------------------------------------------------------------------------- #
# asset-gated real-backend smoke: skipped unless faster-whisper AND
# openai-whisper are actually installed.
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
        result = transcriber(silence, 16000)
        assert isinstance(result, str)
