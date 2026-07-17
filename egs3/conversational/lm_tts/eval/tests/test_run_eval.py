"""Tests for the eval battery orchestrator and report writer (Task 8): the
module that ties every prior task's building block together into one
per-record row, pools rows into a run-level aggregate, and renders a
Markdown comparison report across runs.

Everything heavy (diarization, ASR, speaker embedding, UTMOS) is injected
via ``EvalDeps`` and scripted with fakes keyed by wav path - these tests
never touch pyannote/whisper/WavLM/SpeechMOS. Only the PURE downstream
logic (``eval.metrics.wer.cpwer``/``wer_concat``,
``eval.metrics.simo.segment_similarities``/``cluster_cross_similarity``,
``eval.diarize.purity``, ``eval.asr.assign_words``) runs for real, against
small synthetic wavs, so the two-tone-by-sign trick from
``test_simo_utmos.py`` makes speaker/cluster assignment exact and
hand-checkable.
"""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.asr import Word
from eval.diarize import DiarSegment
from eval.metrics.wer import ErrorCounts, cpwer, wer_concat
from eval.run_eval import EvalDeps, aggregate, evaluate_record, run_battery

_SR = 16000


# ---------------------------------------------------------------------------
# synthetic wav + fake-dep helpers
# ---------------------------------------------------------------------------


def _write_two_tone_wav(path: Path, sign_a: float, sign_b: float) -> str:
    """4s wav: [0,2) constant ``sign_a``, [2,4) constant ``sign_b`` - the
    "positive/negative tone" trick from ``test_simo_utmos.py`` so a
    sign-keyed fake ``embed_fn`` gives exact, hand-checkable cosine
    similarities without paying for a real speaker-embedding model.
    """
    seg_a = np.full(2 * _SR, sign_a, dtype=np.float32)
    seg_b = np.full(2 * _SR, sign_b, dtype=np.float32)
    audio = np.concatenate([seg_a, seg_b])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, _SR)
    return str(path)


def _fake_embed_fn():
    """One-hot-by-sign: [1, 0] for mean-positive audio, [0, 1] otherwise -
    matches ``test_simo_utmos.py``'s fake so cosine similarity is exactly
    1.0 for a correct match and 0.0 for a wrong one.
    """

    def fn(audio: np.ndarray) -> np.ndarray:
        if audio.mean() > 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    return fn


def _make_fake_diarize(segments_by_path: dict[str, list[DiarSegment]]):
    def fn(wav_path: str) -> list[DiarSegment]:
        return segments_by_path[str(wav_path)]

    return fn


def _make_fake_transcribe(transcripts_by_path: dict[str, tuple[str, list[Word]]]):
    def fn(wav_path: str) -> tuple[str, list[Word]]:
        return transcripts_by_path[str(wav_path)]

    return fn


def _make_fake_utmos(scores_by_path: dict[str, float]):
    def fn(wav_path: str) -> float:
        return scores_by_path[str(wav_path)]

    return fn


# ---------------------------------------------------------------------------
# fixture: two manifest entries (one Set A, one Set B) + generated/gt wavs
# ---------------------------------------------------------------------------


@pytest.fixture
def battery_fixture(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    refs_dir = tmp_path / "refs"

    # --- Set A record: two speakers, perfectly-diarized 2-cluster gen wav.
    gen_a = _write_two_tone_wav(wav_dir / "ex_a.wav", 0.1, -0.1)
    gt_a = _write_two_tone_wav(tmp_path / "gt_a.wav", 0.1, -0.1)
    ref_spk1 = _write_two_tone_wav(refs_dir / "spk1.wav", 0.1, 0.1)
    ref_spk2 = _write_two_tone_wav(refs_dir / "spk2.wav", -0.1, -0.1)
    (wav_dir / "ex_a.json").write_text(
        json.dumps({"example_id": "ex_a", "has_audio": True}), encoding="utf-8"
    )

    entry_a = {
        "example_id": "ex_a",
        "set": "sssd",
        "system": "sys",
        "caption": "cap",
        "gt_wav": gt_a,
        "turns": [
            {"speaker": "spk1", "start": 0.0, "end": 2.0, "text": "hello there"},
            {"speaker": "spk2", "start": 2.0, "end": 4.0, "text": "good morning"},
        ],
        "speakers": ["spk1", "spk2"],
        "ref_wavs": {"spk1": ref_spk1, "spk2": ref_spk2},
    }

    # --- Set B record: no speaker labels; gt_wav distinct from gen wav so
    # sim_cross_gt (mode == "generated") is non-trivial.
    gen_b = _write_two_tone_wav(wav_dir / "ex_b.wav", 0.2, -0.2)
    gt_b = _write_two_tone_wav(tmp_path / "gt_b.wav", 0.2, -0.2)
    (wav_dir / "ex_b.json").write_text(
        json.dumps({"example_id": "ex_b", "has_audio": True}), encoding="utf-8"
    )

    entry_b = {
        "example_id": "ex_b",
        "set": "sft",
        "system": "sys",
        "caption": "cap",
        "gt_wav": gt_b,
        "turns": [
            {"speaker": None, "start": None, "end": None, "text": "one two three"},
            {"speaker": None, "start": None, "end": None, "text": "four five"},
        ],
        "speakers": None,
        "ref_wavs": None,
    }

    # --- error record: json marker already carries a generation error, and
    # no wav was ever written - the row must be captured as errored without
    # raising out of run_battery.
    (wav_dir / "ex_err.json").write_text(
        json.dumps({"example_id": "ex_err", "error": "RuntimeError: boom"}),
        encoding="utf-8",
    )
    entry_err = {
        "example_id": "ex_err",
        "set": "sssd",
        "system": "sys",
        "caption": "cap",
        "gt_wav": gt_a,
        "turns": [{"speaker": "spk1", "start": 0.0, "end": 1.0, "text": "x"}],
        "speakers": ["spk1"],
        "ref_wavs": {"spk1": ref_spk1},
    }

    segments_by_path = {
        gen_a: [DiarSegment(0.0, 2.0, "A"), DiarSegment(2.0, 4.0, "B")],
        gt_a: [DiarSegment(0.0, 2.0, "A"), DiarSegment(2.0, 4.0, "B")],
        gen_b: [DiarSegment(0.0, 2.0, "X"), DiarSegment(2.0, 4.0, "Y")],
        gt_b: [DiarSegment(0.0, 2.0, "P"), DiarSegment(2.0, 4.0, "Q")],
    }
    transcripts_by_path = {
        gen_a: (
            "hello there good morning",
            [
                Word("hello", 0.0, 0.5),
                Word("there", 0.5, 1.0),
                Word("good", 2.0, 2.5),
                Word("morning", 2.5, 3.0),
            ],
        ),
        gt_a: (
            "hello there good morning",
            [
                Word("hello", 0.0, 0.5),
                Word("there", 0.5, 1.0),
                Word("good", 2.0, 2.5),
                Word("morning", 2.5, 3.0),
            ],
        ),
        gen_b: ("one two three four five", []),
        gt_b: ("one two three four five", []),
    }
    utmos_by_path = {gen_a: 3.5, gt_a: 4.0, gen_b: 3.0, gt_b: 3.2}

    deps = EvalDeps(
        diarize_fn=_make_fake_diarize(segments_by_path),
        transcribe_fn=_make_fake_transcribe(transcripts_by_path),
        embed_fn=_fake_embed_fn(),
        utmos_fn=_make_fake_utmos(utmos_by_path),
    )

    return {
        "wav_dir": wav_dir,
        "entries": [entry_a, entry_b, entry_err],
        "entry_a": entry_a,
        "entry_b": entry_b,
        "entry_err": entry_err,
        "deps": deps,
        "gen_a": gen_a,
        "gen_b": gen_b,
        "gt_a": gt_a,
        "gt_b": gt_b,
    }


# ---------------------------------------------------------------------------
# evaluate_record: Set A, generated mode
# ---------------------------------------------------------------------------


class TestEvaluateRecordSetAGenerated:
    def test_perfect_alignment_row(self, battery_fixture):
        entry = battery_fixture["entry_a"]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, battery_fixture["gen_a"], deps)

        assert row["example_id"] == "ex_a"
        assert row["error"] is None
        assert row["n_clusters"] == 2

        expected_wer = wer_concat(
            ["hello there", "good morning"], "hello there good morning"
        )
        assert row["wer_concat_counts"] == {
            "hits": expected_wer.hits,
            "substitutions": expected_wer.substitutions,
            "deletions": expected_wer.deletions,
            "insertions": expected_wer.insertions,
        }

        expected_cp = cpwer(
            {"spk1": "hello there", "spk2": "good morning"},
            {"A": "hello there", "B": "good morning"},
        )
        assert row["cpwer_counts"] == {
            "hits": expected_cp.counts.hits,
            "substitutions": expected_cp.counts.substitutions,
            "deletions": expected_cp.counts.deletions,
            "insertions": expected_cp.counts.insertions,
        }
        assert row["cpwer_mapping"] == expected_cp.mapping
        assert row["cpwer_mapping"] == {"A": "spk1", "B": "spk2"}

        assert row["sim_own_mean"] == pytest.approx(1.0)
        assert row["sim_margin_mean"] == pytest.approx(1.0)
        assert row["mapping_disagrees"] is False

        assert row["utmos"] == pytest.approx(3.5)
        assert row["duration_s"] == pytest.approx(4.0)

        # not an anchor row and not Set B: both anchor/cross-gt fields null.
        assert row["purity_gt"] is None
        assert row["sim_cross_gt"] is None

    def test_mapping_disagrees_true_when_cpwer_and_sim_pick_different_speaker(
        self, battery_fixture
    ):
        # Text content says A=spk2's words, B=spk1's words (cpWER will map
        # A->spk2, B->spk1 by content), but the AUDIO tone still says
        # A sounds like spk1 and B sounds like spk2 (sim picks A->spk1,
        # B->spk2) - the two mappings disagree.
        entry = dict(battery_fixture["entry_a"])
        entry["turns"] = [
            {"speaker": "spk2", "start": 0.0, "end": 2.0, "text": "hello there"},
            {"speaker": "spk1", "start": 2.0, "end": 4.0, "text": "good morning"},
        ]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, battery_fixture["gen_a"], deps)

        assert row["cpwer_mapping"] == {"A": "spk2", "B": "spk1"}
        assert row["mapping_disagrees"] is True

    def test_mapping_disagrees_ignores_clusters_sim_could_not_embed(self, tmp_path):
        # Three speakers/clusters: A and B are normal (>= min_sec, so both
        # cpWER and sim map them); C is a 0.3s sliver - assign_words still
        # attributes text to it (assign_words has no duration floor), so
        # cpWER maps it too, but segment_similarities' min_sec=1.0 default
        # excludes it from embedding entirely, so sim.assignment omits it.
        # cpWER and sim AGREE on the two clusters they both cover (A, B) -
        # the extra "C" key that only cpWER has must not, by itself, count
        # as a disagreement.
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        refs_dir = tmp_path / "refs"

        seg_a = np.full(2 * _SR, 0.1, dtype=np.float32)
        seg_b = np.full(2 * _SR, -0.1, dtype=np.float32)
        seg_c = np.full(int(0.3 * _SR), 0.1, dtype=np.float32)
        gen_wav = wav_dir / "ex_c.wav"
        sf.write(str(gen_wav), np.concatenate([seg_a, seg_b, seg_c]), _SR)
        gen_wav = str(gen_wav)

        ref_spk1 = _write_two_tone_wav(refs_dir / "spk1.wav", 0.1, 0.1)
        ref_spk2 = _write_two_tone_wav(refs_dir / "spk2.wav", -0.1, -0.1)
        # spk3's turn spans [4.0, 4.3) - needs a ref wav that actually
        # covers that span, unlike the 4s two-tone helper.
        refs_dir.mkdir(parents=True, exist_ok=True)
        ref_spk3 = str(refs_dir / "spk3.wav")
        sf.write(ref_spk3, np.full(5 * _SR, 0.1, dtype=np.float32), _SR)

        entry = {
            "example_id": "ex_c",
            "set": "sssd",
            "system": "sys",
            "caption": "cap",
            "gt_wav": gen_wav,
            "turns": [
                {"speaker": "spk1", "start": 0.0, "end": 2.0, "text": "hello there"},
                {"speaker": "spk2", "start": 2.0, "end": 4.0, "text": "good morning"},
                {"speaker": "spk3", "start": 4.0, "end": 4.3, "text": "nice day"},
            ],
            "speakers": ["spk1", "spk2", "spk3"],
            "ref_wavs": {"spk1": ref_spk1, "spk2": ref_spk2, "spk3": ref_spk3},
        }

        segments = [
            DiarSegment(0.0, 2.0, "A"),
            DiarSegment(2.0, 4.0, "B"),
            DiarSegment(4.0, 4.3, "C"),
        ]
        words = [
            Word("hello", 0.0, 0.5),
            Word("there", 0.5, 1.0),
            Word("good", 2.0, 2.5),
            Word("morning", 2.5, 3.0),
            Word("nice", 4.0, 4.1),
            Word("day", 4.1, 4.2),
        ]
        deps = EvalDeps(
            diarize_fn=lambda wav: segments,
            transcribe_fn=lambda wav: ("hello there good morning nice day", words),
            embed_fn=_fake_embed_fn(),
            utmos_fn=lambda wav: 3.0,
        )

        row = evaluate_record(entry, gen_wav, deps)

        assert row["error"] is None
        assert row["cpwer_mapping"] == {"A": "spk1", "B": "spk2", "C": "spk3"}
        # sim never saw a >= min_sec segment for "C", so its assignment
        # only covers A/B - and on those two, sim and cpWER agree.
        assert row["mapping_disagrees"] is False


# ---------------------------------------------------------------------------
# evaluate_record: Set A, anchor mode
# ---------------------------------------------------------------------------


class TestEvaluateRecordSetAAnchor:
    def test_anchor_mode_fills_purity_gt(self, battery_fixture):
        entry = battery_fixture["entry_a"]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, battery_fixture["gt_a"], deps, mode="anchor")

        assert row["error"] is None
        assert row["purity_gt"] == pytest.approx(1.0)
        # anchor mode still computes the rest of the Set A battery.
        assert row["cpwer_mapping"] == {"A": "spk1", "B": "spk2"}


# ---------------------------------------------------------------------------
# evaluate_record: Set B (sft)
# ---------------------------------------------------------------------------


class TestEvaluateRecordSetB:
    def test_generated_mode_null_cpwer_nonnull_sim_cross_gt(self, battery_fixture):
        entry = battery_fixture["entry_b"]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, battery_fixture["gen_b"], deps, mode="generated")

        assert row["error"] is None
        assert row["cpwer_counts"] is None
        assert row["cpwer_mapping"] is None
        assert row["sim_own_mean"] is None
        assert row["sim_margin_mean"] is None
        assert row["mapping_disagrees"] is None
        assert row["sim_cross_gt"] == pytest.approx(1.0)

        expected_wer = wer_concat(
            ["one two three", "four five"], "one two three four five"
        )
        assert row["wer_concat_counts"]["hits"] == expected_wer.hits

    def test_anchor_mode_sim_cross_gt_stays_null(self, battery_fixture):
        entry = battery_fixture["entry_b"]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, battery_fixture["gt_b"], deps, mode="anchor")

        assert row["error"] is None
        assert row["sim_cross_gt"] is None


# ---------------------------------------------------------------------------
# evaluate_record: per-record error handling
# ---------------------------------------------------------------------------


class TestEvaluateRecordErrorHandling:
    def test_missing_wav_captured_as_row_error_not_raised(self, battery_fixture, tmp_path):
        entry = battery_fixture["entry_a"]
        deps = battery_fixture["deps"]

        row = evaluate_record(entry, str(tmp_path / "does_not_exist.wav"), deps)

        assert row["error"] is not None
        assert row["example_id"] == "ex_a"

    def test_zero_segments_captured_as_row_error(self, battery_fixture):
        entry = battery_fixture["entry_a"]
        deps = EvalDeps(
            diarize_fn=lambda wav: [],
            transcribe_fn=battery_fixture["deps"].transcribe_fn,
            embed_fn=battery_fixture["deps"].embed_fn,
            utmos_fn=battery_fixture["deps"].utmos_fn,
        )

        row = evaluate_record(entry, battery_fixture["gen_a"], deps)

        assert row["error"] is not None
        assert row["n_clusters"] == 0


# ---------------------------------------------------------------------------
# run_battery: orchestration over a manifest, wav dir, and mode
# ---------------------------------------------------------------------------


class TestRunBattery:
    def test_generated_mode_reads_wav_dir_and_captures_marker_error(
        self, battery_fixture
    ):
        rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="generated",
        )

        by_id = {r["example_id"]: r for r in rows}
        assert set(by_id) == {"ex_a", "ex_b", "ex_err"}

        assert by_id["ex_a"]["error"] is None
        assert by_id["ex_a"]["cpwer_mapping"] == {"A": "spk1", "B": "spk2"}

        assert by_id["ex_b"]["error"] is None
        assert by_id["ex_b"]["sim_cross_gt"] == pytest.approx(1.0)

    def test_malformed_marker_json_captured_as_row_error_others_survive(
        self, battery_fixture
    ):
        # A marker that is present but not valid JSON - e.g. a process
        # killed mid-write, before the real generator's atomic
        # write-then-os.replace - must not raise past this record and
        # abort the whole run.
        (battery_fixture["wav_dir"] / "ex_b.json").write_text(
            "{not valid json", encoding="utf-8"
        )

        rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="generated",
        )

        by_id = {r["example_id"]: r for r in rows}
        assert set(by_id) == {"ex_a", "ex_b", "ex_err"}

        # the malformed-marker record is captured as an errored row...
        assert by_id["ex_b"]["error"] is not None

        # ...but the run completes and every other row is untouched.
        assert by_id["ex_a"]["error"] is None
        assert by_id["ex_a"]["cpwer_mapping"] == {"A": "spk1", "B": "spk2"}
        assert by_id["ex_err"]["error"] is not None
        assert "boom" in by_id["ex_err"]["error"]

        # error record: json marker already carried "error" - run_battery
        # must not crash, and must propagate that error onto the row
        # without ever calling a (nonexistent) wav through the fakes.
        assert by_id["ex_err"]["error"] is not None
        assert "boom" in by_id["ex_err"]["error"]

    def test_anchor_mode_scores_gt_wav(self, battery_fixture):
        rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="anchor",
        )

        by_id = {r["example_id"]: r for r in rows}
        assert by_id["ex_a"]["error"] is None
        assert by_id["ex_a"]["purity_gt"] == pytest.approx(1.0)
        # anchor mode scores gt_wav directly - it does not consult the
        # (possibly error-carrying) generated-wav json marker at all.
        assert by_id["ex_err"]["error"] is None


# ---------------------------------------------------------------------------
# aggregate: pooled ErrorCounts, means over non-null values, error count
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_pools_wer_counts_not_averages(self):
        rows = [
            {
                "wer_concat_counts": {
                    "hits": 10,
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                },
                "cpwer_counts": None,
                "sim_own_mean": None,
                "sim_margin_mean": None,
                "sim_cross_gt": None,
                "utmos": None,
                "mapping_disagrees": None,
                "error": None,
            },
            {
                "wer_concat_counts": {
                    "hits": 0,
                    "substitutions": 1,
                    "deletions": 1,
                    "insertions": 0,
                },
                "cpwer_counts": None,
                "sim_own_mean": None,
                "sim_margin_mean": None,
                "sim_cross_gt": None,
                "utmos": None,
                "mapping_disagrees": None,
                "error": None,
            },
        ]

        agg = aggregate(rows)

        # pooled: 2 errors / 12 ref_words, NOT mean(0.0, 1.0) = 0.5.
        assert agg["wer_concat"]["counts"] == {
            "hits": 10,
            "substitutions": 1,
            "deletions": 1,
            "insertions": 0,
        }
        assert agg["wer_concat"]["wer"] == pytest.approx(2 / 12)

    def test_means_ignore_nulls(self):
        rows = [
            {"sim_own_mean": 0.8, "utmos": 3.0, "mapping_disagrees": False},
            {"sim_own_mean": None, "utmos": 4.0, "mapping_disagrees": None},
            {"sim_own_mean": 0.6, "utmos": None, "mapping_disagrees": True},
        ]
        for r in rows:
            r.setdefault("wer_concat_counts", None)
            r.setdefault("cpwer_counts", None)
            r.setdefault("sim_margin_mean", None)
            r.setdefault("sim_cross_gt", None)
            r.setdefault("error", None)

        agg = aggregate(rows)

        assert agg["sim_own_mean"] == pytest.approx((0.8 + 0.6) / 2)
        assert agg["utmos_mean"] == pytest.approx((3.0 + 4.0) / 2)
        assert agg["mapping_disagreement_rate"] == pytest.approx(1 / 2)

    def test_counts_row_errors(self):
        rows = [
            {"error": "boom", "wer_concat_counts": None, "cpwer_counts": None,
             "sim_own_mean": None, "sim_margin_mean": None, "sim_cross_gt": None,
             "utmos": None, "mapping_disagrees": None},
            {"error": None, "wer_concat_counts": None, "cpwer_counts": None,
             "sim_own_mean": None, "sim_margin_mean": None, "sim_cross_gt": None,
             "utmos": None, "mapping_disagrees": None},
        ]
        agg = aggregate(rows)
        assert agg["n_err"] == 1
        assert agg["n_rows"] == 2

    def test_all_null_wer_does_not_raise_on_zero_ref_words(self):
        rows = [
            {"error": "boom", "wer_concat_counts": None, "cpwer_counts": None,
             "sim_own_mean": None, "sim_margin_mean": None, "sim_cross_gt": None,
             "utmos": None, "mapping_disagrees": None},
        ]
        agg = aggregate(rows)
        assert agg["wer_concat"]["wer"] is None
        assert agg["cpwer"]["wer"] is None

    def test_aggregate_skips_nan_rows_in_means(self):
        # A single NaN per-row value must not poison the pooled mean.
        # Regression: one NaN sim_margin_mean (of 35 numeric rows)
        # suppressed the whole sim_margin_mean aggregate to NaN.
        rows = [
            {"error": None, "wer_concat_counts": None, "cpwer_counts": None,
             "sim_own_mean": 0.8, "sim_margin_mean": float("nan"),
             "sim_cross_gt": None, "utmos": 3.0, "mapping_disagrees": None},
            {"error": None, "wer_concat_counts": None, "cpwer_counts": None,
             "sim_own_mean": 0.6, "sim_margin_mean": 0.2,
             "sim_cross_gt": None, "utmos": float("nan"), "mapping_disagrees": None},
        ]
        agg = aggregate(rows)
        assert agg["sim_margin_mean"] == pytest.approx(0.2)
        assert agg["utmos_mean"] == pytest.approx(3.0)
        assert agg["sim_own_mean"] == pytest.approx((0.8 + 0.6) / 2)

    def test_end_to_end_aggregate_from_run_battery(self, battery_fixture):
        rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="generated",
        )
        agg = aggregate(rows)

        assert agg["n_rows"] == 3
        assert agg["n_err"] == 1
        # ex_a's cpWER is perfect (0 errors / 4 ref words); ex_b/ex_err
        # contribute no cpwer_counts (Set B null / errored), so pooled
        # cpWER equals ex_a's alone.
        assert agg["cpwer"]["wer"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI: import hygiene + writes results.json
# ---------------------------------------------------------------------------


def test_importing_eval_run_eval_and_report_does_not_load_heavy_deps():
    """Subprocess-based, matching ``eval/tests/test_generate_espnet.py``'s
    idiom: importing these modules must never pull in torch/transformers/
    pyannote at module scope - only inside ``EvalDeps`` construction (the
    real, non-faked path) or ``main``.

    We compare ``sys.modules`` before/after the import and flag only
    NEWLY-added heavy modules: pyannote.audio ships a legacy namespace
    ``.pth`` that does ``sys.modules.setdefault('pyannote', ...)`` at
    interpreter startup, so a bare empty ``pyannote`` namespace stub is
    present before any import. That stub is not an eager heavy-dep load;
    the before/after diff correctly ignores it while still catching a real
    ``import pyannote.audio`` (or torch/transformers) at module scope.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "_before = set(sys.modules); "
            "import eval.run_eval, eval.report; "
            "heavy = [m for m in set(sys.modules) - _before "
            "if m.split('.')[0] in ('torch', 'transformers', 'pyannote')]; "
            "assert not heavy, heavy",
        ],
        env=__import__("os").environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import eval.run_eval/eval.report pulled in heavy deps "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


class TestCli:
    def test_main_writes_results_json_with_rows_and_aggregate(
        self, battery_fixture, tmp_path, monkeypatch
    ):
        import eval.run_eval as run_eval_mod

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                [battery_fixture["entry_a"], battery_fixture["entry_b"]]
            ),
            encoding="utf-8",
        )
        out_path = tmp_path / "results.json"

        monkeypatch.setattr(
            run_eval_mod, "_build_deps", lambda args: battery_fixture["deps"]
        )

        rc = run_eval_mod.main(
            [
                "--manifest",
                str(manifest_path),
                "--wav-dir",
                str(battery_fixture["wav_dir"]),
                "--mode",
                "generated",
                "--out",
                str(out_path),
            ]
        )

        assert rc == 0
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert "rows" in result and "aggregate" in result
        assert {r["example_id"] for r in result["rows"]} == {"ex_a", "ex_b"}
        assert result["aggregate"]["n_rows"] == 2


# ---------------------------------------------------------------------------
# report.write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_table_and_windows_csv_and_best_worst(self, battery_fixture, tmp_path):
        from eval.report import write_report

        rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="generated",
        )
        agg = aggregate(rows)
        results = {"rows": rows, "aggregate": agg}

        run_a_path = tmp_path / "sssd_vllm_results.json"
        run_a_path.write_text(json.dumps(results), encoding="utf-8")

        anchor_rows = run_battery(
            battery_fixture["entries"],
            battery_fixture["wav_dir"],
            battery_fixture["deps"],
            mode="anchor",
        )
        anchor_agg = aggregate(anchor_rows)
        anchor_results = {"rows": anchor_rows, "aggregate": anchor_agg}
        run_b_path = tmp_path / "sssd_gt_anchor_results.json"
        run_b_path.write_text(json.dumps(anchor_results), encoding="utf-8")

        out_md = tmp_path / "report.md"
        write_report(
            {"sssd_vllm": str(run_a_path), "sssd_gt_anchor": str(run_b_path)},
            out_md,
        )

        assert out_md.exists()
        text = out_md.read_text(encoding="utf-8")
        assert "sssd_vllm" in text
        assert "sssd_gt_anchor" in text
        for col in (
            "WER_concat",
            "cpWER",
            "SIM_own",
            "SIM_margin",
            "sim_cross_gt",
            "UTMOS",
            "n_err",
        ):
            assert col in text

        windows_a = tmp_path / "sssd_vllm_windows.csv"
        windows_b = tmp_path / "sssd_gt_anchor_windows.csv"
        assert windows_a.exists()
        assert windows_b.exists()

        with windows_a.open(encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert {r["example_id"] for r in csv_rows} == {"ex_a", "ex_b", "ex_err"}

        # best/worst-5-by-cpWER section present, naming an example_id.
        assert "ex_a" in text


def test_importing_eval_report_alone_does_not_load_heavy_deps():
    stale = [
        name
        for name in sys.modules
        if name == "eval.report" or name.split(".")[0] in ("torch", "transformers")
    ]
    for name in stale:
        del sys.modules[name]

    importlib.import_module("eval.report")

    loaded_heavy = [
        name for name in sys.modules if name.split(".")[0] in ("torch", "transformers")
    ]
    assert not loaded_heavy, f"heavy deps loaded on import: {loaded_heavy}"
