"""Tests for the VERSA wrapper's score aggregation."""

import json
import logging
import re

import pytest

from src.metrics.versa import VersaMetric


def _write_results(tmp_path, rows):
    path = tmp_path / "result.json"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_aggregate_pools_error_counts_instead_of_averaging(tmp_path):
    """WER pools errors over reference words, never averages per-utterance WERs.

    Utterance 1: D=1, I=0, S=0, C=1   -> 1 error out of 2 ref words  -> 50.00%
    Utterance 2: D=0, I=0, S=1, C=97  -> 1 error out of 98 ref words ->  1.02%
    Pooled:      D=1, I=0, S=1, C=98
      N      = D + S + C = 1 + 1 + 98 = 100
      errors = D + S + I = 1 + 1 + 0  = 2
      WER    = 2 / 100 * 100 = 2.00%
    Averaged (wrong):                    25.51%

    There are no insertions here, so this fixture cannot tell the reference
    denominator apart from the alignment-length one. See
    test_aggregate_excludes_insertions_from_the_denominator for that.
    """
    rows = [
        {
            "fwhisper_wer_delete": 1,
            "fwhisper_wer_insert": 0,
            "fwhisper_wer_replace": 0,
            "fwhisper_wer_equal": 1,
        },
        {
            "fwhisper_wer_delete": 0,
            "fwhisper_wer_insert": 0,
            "fwhisper_wer_replace": 1,
            "fwhisper_wer_equal": 97,
        },
    ]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    assert scores["fwhisper_wer"] == pytest.approx(2.0)


def test_aggregate_still_returns_component_averages(tmp_path):
    rows = [
        {
            "fwhisper_wer_delete": 1,
            "fwhisper_wer_insert": 0,
            "fwhisper_wer_replace": 0,
            "fwhisper_wer_equal": 1,
        },
        {
            "fwhisper_wer_delete": 0,
            "fwhisper_wer_insert": 0,
            "fwhisper_wer_replace": 1,
            "fwhisper_wer_equal": 97,
        },
    ]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    assert scores["fwhisper_wer_equal"] == pytest.approx(49.0)
    assert scores["fwhisper_wer_delete"] == pytest.approx(0.5)


def test_aggregate_pools_cer_too(tmp_path):
    """CER pools the same way WER does, on the reference-length denominator.

    Utterance 1: D=2, I=0, S=0, C=8   -> ref  10, errors 2
    Utterance 2: D=0, I=3, S=0, C=87  -> ref  87, errors 3
    Pooled:      D=2, I=3, S=0, C=95
      N      = D + S + C = 2 + 0 + 95 = 97
      errors = D + S + I = 2 + 0 + 3  = 5
      CER    = 5 / 97 * 100 = 5.154639... -> rounded to 5.1546
    """
    rows = [
        {
            "fwhisper_cer_delete": 2,
            "fwhisper_cer_insert": 0,
            "fwhisper_cer_replace": 0,
            "fwhisper_cer_equal": 8,
        },
        {
            "fwhisper_cer_delete": 0,
            "fwhisper_cer_insert": 3,
            "fwhisper_cer_replace": 0,
            "fwhisper_cer_equal": 87,
        },
    ]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    # _aggregate rounds to 4 decimals, so this compares exactly.
    assert scores["fwhisper_cer"] == pytest.approx(5.1546)


def test_aggregate_excludes_insertions_from_the_denominator(tmp_path):
    """The WER denominator is the reference length, not the alignment length.

    Insertions are errors, so they count in the numerator, but they are not
    reference tokens, so they must NOT count in the denominator. VERSA's own
    fwhisper_wer asserts ``delete + replace + equal == len(ref_words)``.

    Pooled counts here: D=3, I=50, S=7, C=40.

      correct:  N      = D + S + C = 3 + 7 + 40 = 50
                errors = D + S + I = 3 + 7 + 50 = 60
                WER    = 60 / 50 * 100 = 120.0%

      wrong:    total  = D + I + S + C = 100
                errors = total - C     = 60
                WER    = 60 / 100 * 100 = 60.0%

    A rate above 100% is correct, not a bug: the hypothesis can contain more
    inserted words than the reference has words in total.
    """
    rows = [
        {
            "fwhisper_wer_delete": 2,
            "fwhisper_wer_insert": 30,
            "fwhisper_wer_replace": 3,
            "fwhisper_wer_equal": 15,
        },
        {
            "fwhisper_wer_delete": 1,
            "fwhisper_wer_insert": 20,
            "fwhisper_wer_replace": 4,
            "fwhisper_wer_equal": 25,
        },
    ]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    assert scores["fwhisper_wer"] == pytest.approx(120.0)


def test_aggregate_skips_metric_when_reference_is_empty(tmp_path):
    """An empty reference must not raise; the pooled key is simply omitted.

    With the corrected denominator, N = D + S + C is zero whenever the
    reference is empty, so the guard is what stands between an
    empty-transcript row and a ZeroDivisionError in the measure stage.
    Insertions alone (I=5) do not make the rate definable.
    """
    rows = [
        {
            "fwhisper_wer_delete": 0,
            "fwhisper_wer_insert": 5,
            "fwhisper_wer_replace": 0,
            "fwhisper_wer_equal": 0,
        },
    ]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    assert "fwhisper_wer" not in scores
    # The per-op component averages are still reported.
    assert scores["fwhisper_wer_insert"] == pytest.approx(5.0)


def test_aggregate_leaves_non_edit_metrics_alone(tmp_path):
    rows = [{"utmos": 4.0}, {"utmos": 4.5}]
    scores = VersaMetric._aggregate(_write_results(tmp_path, rows))
    assert scores["utmos"] == pytest.approx(4.25)
    assert "fwhisper_wer" not in scores


def test_find_prefix_requires_all_four_ops():
    partial = {"fwhisper_wer_delete": 1, "fwhisper_wer_equal": 9}
    assert VersaMetric._find_prefix(partial, "wer") is None

    complete = {
        "fwhisper_wer_delete": 1,
        "fwhisper_wer_insert": 0,
        "fwhisper_wer_replace": 0,
        "fwhisper_wer_equal": 9,
    }
    assert VersaMetric._find_prefix(complete, "wer") == "fwhisper_wer_"


def test_summarize_reports_pooled_wer_exactly_once(caplog):
    """The pooled ``fwhisper_wer`` scalar must be reported exactly once.

    It belongs in the labeled WER components block, not a second time in
    the unlabeled main section. Before the fix, ``main_keys`` missed the
    pooled key (it lacks the trailing underscore that ``wer_prefix``
    carries), so it leaked into the generic
    ``f"  {k:<25s} {scores[k]:.4f}"`` line as well.
    """
    scores = {
        "fwhisper_wer_delete": 1.0,
        "fwhisper_wer_insert": 0.0,
        "fwhisper_wer_replace": 0.0,
        "fwhisper_wer_equal": 1.0,
        "fwhisper_wer": 50.0,
    }
    with caplog.at_level(logging.INFO, logger="src.metrics.versa"):
        VersaMetric.summarize(scores, test_name="unit-test")

    # Match "fwhisper_wer" only when NOT immediately followed by "_", so the
    # per-op keys (fwhisper_wer_delete, ...) don't count toward this total.
    # The labeled components-block header ("[fwhisper_wer]:") is expected to
    # match once; a second match would mean the pooled key also leaked into
    # the unlabeled main section.
    occurrences = re.findall(r"fwhisper_wer(?!_)", caplog.text)
    assert len(occurrences) == 1
