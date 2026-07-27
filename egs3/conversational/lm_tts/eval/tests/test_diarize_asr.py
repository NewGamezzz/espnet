"""Tests for diarization + ASR wrappers (Task 4).

Covers the two PURE functions only - `assign_words` (word-to-segment
assignment) and `purity` (diarization purity against GT turns) - plus an
import-hygiene check that importing `eval.diarize` / `eval.asr` never
pulls in `pyannote` or `transformers`, since those heavy, GPU-bound
dependencies must load lazily, inside `diarize()` / `transcribe()` only,
so unit tests never trigger a model download. `diarize()` and
`transcribe()` themselves are model-backed wrappers and are not
unit-tested here; Task 8 exercises them end to end.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from eval.asr import Word, assign_words
from eval.diarize import DiarSegment, purity

# ---------------------------------------------------------------------------
# assign_words
# ---------------------------------------------------------------------------


def test_assign_words_midpoint_containment():
    segments = [
        DiarSegment(start=0.0, end=5.0, cluster="A"),
        DiarSegment(start=5.0, end=10.0, cluster="B"),
    ]
    words = [
        Word(text="hi", start=1.0, end=2.0),  # midpoint 1.5 -> A
        Word(text="there", start=6.0, end=7.0),  # midpoint 6.5 -> B
    ]
    assert assign_words(words, segments) == {"A": "hi", "B": "there"}


def test_assign_words_out_of_segment_word_snaps_to_nearest_boundary():
    segments = [
        DiarSegment(start=0.0, end=5.0, cluster="A"),
        DiarSegment(start=5.0, end=10.0, cluster="B"),
        DiarSegment(start=10.0, end=15.0, cluster="C"),
    ]
    # midpoint 20.5 is contained by no segment; nearest-boundary distances
    # are A: 15.5, B: 10.5, C: 5.5 -> snaps to C.
    words = [Word(text="edge", start=20.0, end=21.0)]
    assert assign_words(words, segments) == {"A": "", "B": "", "C": "edge"}


def test_assign_words_empty_cluster_present_with_empty_string():
    segments = [
        DiarSegment(start=0.0, end=5.0, cluster="A"),
        DiarSegment(start=5.0, end=10.0, cluster="B"),
    ]
    words = [Word(text="hi", start=1.0, end=2.0)]
    assert assign_words(words, segments) == {"A": "hi", "B": ""}


def test_assign_words_ordering_preserved_regardless_of_input_order():
    segments = [DiarSegment(start=0.0, end=10.0, cluster="A")]
    # Fed out of time order; output must join in time order, not input order.
    words = [
        Word(text="beta", start=3.0, end=4.0),
        Word(text="hello", start=1.0, end=2.0),
    ]
    assert assign_words(words, segments) == {"A": "hello beta"}


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------


def test_purity_perfect_two_cluster_two_speaker_alignment_is_one():
    segments = [
        DiarSegment(start=0.0, end=5.0, cluster="A"),
        DiarSegment(start=5.0, end=10.0, cluster="B"),
    ]
    gt_turns = [
        {"speaker": "spk1", "start": 0.0, "end": 5.0, "text": "hello"},
        {"speaker": "spk2", "start": 5.0, "end": 10.0, "text": "world"},
    ]
    assert purity(segments, gt_turns) == pytest.approx(1.0)


def test_purity_one_cluster_spanning_both_speakers_fifty_fifty():
    # cluster A overlaps spk1 for 5s and spk2 for 5s: max(5, 5) / (5 + 5).
    segments = [DiarSegment(start=0.0, end=10.0, cluster="A")]
    gt_turns = [
        {"speaker": "spk1", "start": 0.0, "end": 5.0, "text": "hello"},
        {"speaker": "spk2", "start": 5.0, "end": 10.0, "text": "world"},
    ]
    assert purity(segments, gt_turns) == pytest.approx(0.5)


def test_purity_no_overlap_is_zero():
    segments = [DiarSegment(start=0.0, end=5.0, cluster="A")]
    gt_turns = [{"speaker": "spk1", "start": 10.0, "end": 15.0, "text": "hello"}]
    assert purity(segments, gt_turns) == 0.0


def test_purity_weights_by_overlap_time_not_segment_count():
    # cluster A: two segments both fully agreeing with spk1 (8s total),
    # one short segment (2s) that only overlaps spk2. Majority-overlap
    # speaker for A is spk1 (8 > 2), so numerator is 8, denominator is 10.
    segments = [
        DiarSegment(start=0.0, end=4.0, cluster="A"),
        DiarSegment(start=4.0, end=8.0, cluster="A"),
        DiarSegment(start=8.0, end=10.0, cluster="A"),
    ]
    gt_turns = [
        {"speaker": "spk1", "start": 0.0, "end": 8.0, "text": "hello"},
        {"speaker": "spk2", "start": 8.0, "end": 10.0, "text": "world"},
    ]
    assert purity(segments, gt_turns) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# diarize() num_speakers threading (against a fake cached pipeline - the
# module-global cache lets us exercise the wrapper without pyannote)
# ---------------------------------------------------------------------------


def test_diarize_forwards_num_speakers_only_when_set(monkeypatch):
    from eval import diarize as diarize_mod

    calls = []

    class _FakeAnnotation:
        def itertracks(self, yield_label=True):
            return iter(())

    class _FakePipeline:
        def __call__(self, wav_path, **kwargs):
            calls.append((wav_path, kwargs))
            return _FakeAnnotation()

    monkeypatch.setattr(diarize_mod, "_pipeline", _FakePipeline())

    assert diarize_mod.diarize("a.wav", num_speakers=2) == []
    assert diarize_mod.diarize("b.wav") == []
    assert calls == [("a.wav", {"num_speakers": 2}), ("b.wav", {})]


# ---------------------------------------------------------------------------
# import hygiene: importing the wrapper modules must not import the heavy,
# GPU-bound deps - those load lazily, only inside diarize()/transcribe().
# ---------------------------------------------------------------------------


def test_importing_eval_diarize_and_asr_does_not_load_heavy_deps():
    stale = [
        name
        for name in sys.modules
        if name in ("eval.diarize", "eval.asr")
        or name.split(".")[0] in ("pyannote", "transformers")
    ]
    for name in stale:
        del sys.modules[name]

    importlib.import_module("eval.diarize")
    importlib.import_module("eval.asr")

    loaded_heavy = [
        name
        for name in sys.modules
        if name.split(".")[0] in ("pyannote", "transformers")
    ]
    assert not loaded_heavy, f"heavy deps loaded on import: {loaded_heavy}"
