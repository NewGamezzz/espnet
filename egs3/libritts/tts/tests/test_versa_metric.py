"""Tests for the VERSA wrapper's score aggregation."""

import json

import pytest

from src.metrics.versa import VersaMetric


def _write_results(tmp_path, rows):
    path = tmp_path / "result.json"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_aggregate_pools_error_counts_instead_of_averaging(tmp_path):
    """WER must be sum(errors)/sum(total), never the mean of per-utterance WERs.

    Utterance 1: 1 error out of 2 words   -> 50.00% on its own
    Utterance 2: 1 error out of 98 words  ->  1.02% on its own
    Pooled:      2 errors out of 100      ->  2.00%
    Averaged (wrong):                        25.51%
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
    assert scores["fwhisper_cer"] == pytest.approx(5.0)


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
