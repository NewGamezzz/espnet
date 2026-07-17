"""Tests for speaker-similarity metrics and the UTMOS wrapper (Task 5).

Covers the PURE assignment/averaging logic in `eval.metrics.simo` -
`reference_embedding`, `segment_similarities`, `cluster_cross_similarity`
- with a fake `embed_fn` that returns a one-hot vector keyed on which
"tone" (positive- vs negative-amplitude constant segment) the audio came
from, so cosine similarity is exactly 1.0 for a correct speaker/cluster
match and 0.0 for a wrong one. This lets assignment correctness,
`sim_own_mean`, and `margin_mean` be asserted exactly, without paying for
WavLM. `default_embed_fn` (the real WavLM x-vector extractor) and
`utmos()` (the real SpeechMOS model) are model-backed wrappers and are
not functionally unit-tested here - only import hygiene is checked, the
same `sys.modules` pattern `eval/tests/test_diarize_asr.py` uses for
Task 4's `diarize()`/`transcribe()`.
"""

from __future__ import annotations

import importlib
import math
import sys

import numpy as np
import pytest
import soundfile as sf

from eval.diarize import DiarSegment
from eval.metrics.simo import (
    SimResult,
    cluster_cross_similarity,
    reference_embedding,
    segment_similarities,
)

_SR = 16000

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _fake_embed_fn(calls: list[int] | None = None):
    """One-hot-by-sign fake: [1, 0] for mean-positive audio, [0, 1]
    otherwise. Optionally records each call's sample count in `calls`.
    """

    def fn(audio: np.ndarray) -> np.ndarray:
        if calls is not None:
            calls.append(len(audio))
        if audio.mean() > 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    return fn


def _sample_count_embed_fn(received: list[int]):
    def fn(audio: np.ndarray) -> np.ndarray:
        received.append(len(audio))
        return np.zeros(2, dtype=np.float32)

    return fn


def _write_wav(path, audio: np.ndarray, sr: int = _SR) -> str:
    sf.write(str(path), audio, sr)
    return str(path)


# ---------------------------------------------------------------------------
# reference_embedding
# ---------------------------------------------------------------------------


def test_reference_embedding_crops_to_turns_and_concatenates(tmp_path):
    audio = np.linspace(-1.0, 1.0, 10 * _SR, dtype=np.float32)
    wav_path = _write_wav(tmp_path / "ref.wav", audio)

    received: list[int] = []
    turns = [
        {"speaker": "spk1", "start": 0.0, "end": 2.0, "text": "hi"},
        {"speaker": "spk1", "start": 5.0, "end": 8.0, "text": "there"},
    ]

    reference_embedding(wav_path, turns, _sample_count_embed_fn(received), max_sec=30.0)

    # 2s + 3s concatenated = 5s, under the cap.
    assert received == [5 * _SR]


def test_reference_embedding_caps_duration(tmp_path):
    audio = np.linspace(-1.0, 1.0, 10 * _SR, dtype=np.float32)
    wav_path = _write_wav(tmp_path / "ref.wav", audio)

    received: list[int] = []
    turns = [{"speaker": "spk1", "start": 0.0, "end": 8.0, "text": "long"}]

    reference_embedding(wav_path, turns, _sample_count_embed_fn(received), max_sec=3.0)

    assert received == [3 * _SR]


def test_reference_embedding_whole_file_when_turns_none(tmp_path):
    audio = np.zeros(4 * _SR, dtype=np.float32)
    wav_path = _write_wav(tmp_path / "ref.wav", audio)

    received: list[int] = []
    reference_embedding(wav_path, None, _sample_count_embed_fn(received), max_sec=30.0)

    assert received == [4 * _SR]


def test_reference_embedding_none_when_all_turns_out_of_bounds(tmp_path):
    # Every turn span falls past the end of a 4s ref wav -> no valid audio.
    # Must return None WITHOUT embedding an empty clip (which would yield a
    # NaN vector that then poisons similarities and the assignment).
    audio = np.zeros(4 * _SR, dtype=np.float32)
    wav_path = _write_wav(tmp_path / "ref.wav", audio)

    received: list[int] = []
    turns = [{"speaker": "spk1", "start": 10.0, "end": 12.0, "text": "gone"}]
    result = reference_embedding(
        wav_path, turns, _sample_count_embed_fn(received), max_sec=30.0
    )

    assert result is None
    assert received == []  # embed_fn never called on empty audio


# ---------------------------------------------------------------------------
# segment_similarities
# ---------------------------------------------------------------------------


def _two_tone_gen_wav(path):
    """4.5s wav: 0-2s positive tone (-> cluster A), 2-4s negative tone
    (-> cluster B), 4-4.5s a short positive blip (-> cluster C, under
    min_sec so it must be skipped for embedding entirely).
    """
    seg_a = np.full(2 * _SR, 0.1, dtype=np.float32)
    seg_b = np.full(2 * _SR, -0.1, dtype=np.float32)
    seg_c = np.full(int(0.5 * _SR), 0.1, dtype=np.float32)
    audio = np.concatenate([seg_a, seg_b, seg_c])
    return _write_wav(path, audio)


def test_segment_similarities_correct_assignment_and_positive_margin(tmp_path):
    gen_wav = _two_tone_gen_wav(tmp_path / "gen.wav")
    segments = [
        DiarSegment(start=0.0, end=2.0, cluster="A"),
        DiarSegment(start=2.0, end=4.0, cluster="B"),
        DiarSegment(start=4.0, end=4.5, cluster="C"),  # < min_sec: skipped
    ]

    plain = _fake_embed_fn()
    ref_embs = {
        "spk1": plain(np.full(_SR, 0.1, dtype=np.float32)),
        "spk2": plain(np.full(_SR, -0.1, dtype=np.float32)),
    }

    calls: list[int] = []
    result = segment_similarities(
        gen_wav, segments, ref_embs, _fake_embed_fn(calls), min_sec=1.0
    )

    assert isinstance(result, SimResult)
    assert result.assignment == {"A": "spk1", "B": "spk2"}
    assert result.sim_own_mean == pytest.approx(1.0)
    assert result.margin_mean == pytest.approx(1.0)
    assert result.margin_mean > 0
    # Only the two >= min_sec segments (A, B) get embedded; the 0.5s "C"
    # segment is skipped entirely and never reaches embed_fn.
    assert len(calls) == 2


def test_segment_similarities_single_speaker_margin_is_none(tmp_path):
    gen_wav = _two_tone_gen_wav(tmp_path / "gen.wav")
    segments = [
        DiarSegment(start=0.0, end=2.0, cluster="A"),
        DiarSegment(start=2.0, end=4.0, cluster="B"),
    ]

    plain = _fake_embed_fn()
    ref_embs = {"spk1": plain(np.full(_SR, 0.1, dtype=np.float32))}

    result = segment_similarities(gen_wav, segments, ref_embs, _fake_embed_fn())

    assert result.margin_mean is None


def test_segment_similarities_skips_none_reference_speaker(tmp_path):
    # A speaker whose reference crop was degenerate (None embedding) must
    # be excluded, not injected as NaN similarities that bias assignment.
    gen_wav = _two_tone_gen_wav(tmp_path / "gen.wav")
    segments = [
        DiarSegment(start=0.0, end=2.0, cluster="A"),
        DiarSegment(start=2.0, end=4.0, cluster="B"),
    ]

    plain = _fake_embed_fn()
    ref_embs = {
        "spk1": plain(np.full(_SR, 0.1, dtype=np.float32)),
        "spk2": None,  # degenerate reference
    }

    result = segment_similarities(gen_wav, segments, ref_embs, _fake_embed_fn())

    # spk2 is excluded: no pair mentions it, and nothing is NaN.
    assert all(speaker == "spk1" for (_cluster, speaker) in result.sim_matrix)
    assert not math.isnan(result.sim_own_mean)
    assert result.margin_mean is None  # only one usable speaker


# ---------------------------------------------------------------------------
# cluster_cross_similarity
# ---------------------------------------------------------------------------


def test_cluster_cross_similarity_optimal_matching_2x2(tmp_path):
    # gen: cluster A positive, cluster B negative.
    gen_audio = np.concatenate(
        [
            np.full(2 * _SR, 0.1, dtype=np.float32),
            np.full(2 * _SR, -0.1, dtype=np.float32),
        ]
    )
    gen_wav = _write_wav(tmp_path / "gen.wav", gen_audio)
    gen_segments = [
        DiarSegment(start=0.0, end=2.0, cluster="A"),
        DiarSegment(start=2.0, end=4.0, cluster="B"),
    ]

    # gt: cluster P negative, cluster Q positive - deliberately "crossed"
    # relative to gen's sorted order, so only the non-trivial assignment
    # (A<->Q, B<->P) is optimal; the naive sorted-order pairing (A<->P,
    # B<->Q) would score 0.
    gt_audio = np.concatenate(
        [
            np.full(2 * _SR, -0.1, dtype=np.float32),
            np.full(2 * _SR, 0.1, dtype=np.float32),
        ]
    )
    gt_wav = _write_wav(tmp_path / "gt.wav", gt_audio)
    gt_segments = [
        DiarSegment(start=0.0, end=2.0, cluster="P"),
        DiarSegment(start=2.0, end=4.0, cluster="Q"),
    ]

    score = cluster_cross_similarity(
        gen_wav, gen_segments, gt_wav, gt_segments, _fake_embed_fn(), min_sec=1.0
    )

    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# import hygiene: importing the wrappers must not load torch/transformers -
# those load lazily, only inside default_embed_fn()/utmos().
# ---------------------------------------------------------------------------


def test_importing_eval_metrics_simo_and_utmos_does_not_load_heavy_deps():
    stale = [
        name
        for name in sys.modules
        if name in ("eval.metrics.simo", "eval.metrics.utmos")
        or name.split(".")[0] in ("torch", "transformers")
    ]
    for name in stale:
        del sys.modules[name]

    importlib.import_module("eval.metrics.simo")
    importlib.import_module("eval.metrics.utmos")

    loaded_heavy = [
        name for name in sys.modules if name.split(".")[0] in ("torch", "transformers")
    ]
    assert not loaded_heavy, f"heavy deps loaded on import: {loaded_heavy}"
