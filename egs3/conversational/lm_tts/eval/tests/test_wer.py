"""Tests for the eval WER metrics (Task 3): pooled ``ErrorCounts``,
``wer_concat``, and ``cpwer``, matching the "BagPiper Conversational
Baseline Eval" plan's binding rule - WER is computed by pooling I/D/S
error counts and reference word counts across utterances THEN dividing,
never by averaging per-utterance WERs.
"""

from __future__ import annotations

import pytest
from eval.metrics.wer import (
    CpWerResult,
    ErrorCounts,
    count_errors,
    cpwer,
    normalize,
    wer_concat,
)

# ---------------------------------------------------------------------------
# ErrorCounts: properties and pooling
# ---------------------------------------------------------------------------


def test_error_counts_wer_math():
    counts = ErrorCounts(hits=8, substitutions=1, deletions=1, insertions=0)
    assert counts.ref_words == 10
    assert counts.errors == 2
    assert counts.wer == pytest.approx(0.2)


def test_error_counts_wer_raises_on_zero_ref_words():
    counts = ErrorCounts(hits=0, substitutions=0, deletions=0, insertions=3)
    with pytest.raises(ValueError):
        counts.wer


def test_error_counts_add_is_fieldwise():
    a = ErrorCounts(hits=1, substitutions=2, deletions=3, insertions=4)
    b = ErrorCounts(hits=5, substitutions=6, deletions=7, insertions=8)
    total = a + b
    assert total == ErrorCounts(hits=6, substitutions=8, deletions=10, insertions=12)


def test_pooled_counts_not_averaged():
    a = count_errors("a b c d e f g h i j", "a b c d e f g h i j")  # 0/10
    b = count_errors("x y", "p q")  # 2/2
    pooled = a + b
    assert pooled.wer == pytest.approx(2 / 12)  # NOT (0.0 + 1.0) / 2


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_folds_case_and_punctuation():
    assert normalize("Hello, THERE!") == normalize("hello there")


def test_normalize_produces_matching_counts():
    counts = count_errors("Hello, THERE!", "hello there")
    assert counts.errors == 0
    assert counts.ref_words == 2


# ---------------------------------------------------------------------------
# count_errors
# ---------------------------------------------------------------------------


def test_count_errors_empty_normalized_ref_raises():
    with pytest.raises(ValueError):
        count_errors("", "something")


def test_count_errors_ref_that_normalizes_to_empty_raises():
    with pytest.raises(ValueError):
        count_errors("   ", "something")


# ---------------------------------------------------------------------------
# wer_concat
# ---------------------------------------------------------------------------


def test_wer_concat_joins_turns_with_space():
    counts = wer_concat(["hello there", "good morning"], "hello there good morning")
    assert counts.errors == 0
    assert counts.ref_words == 4


def test_wer_concat_counts_errors_against_concatenated_ref():
    counts = wer_concat(["good day", "kind sir"], "good day kind ma'am")
    assert counts.ref_words == 4
    assert counts.errors == 1
    assert counts.substitutions == 1


# ---------------------------------------------------------------------------
# cpwer
# ---------------------------------------------------------------------------


def test_cpwer_finds_speaker_permutation():
    refs = {"s1": "hello there", "s2": "good morning to you"}
    hyps = {"c0": "good morning to you", "c1": "hello there"}
    r = cpwer(refs, hyps)
    assert r.counts.errors == 0
    assert r.mapping == {"c0": "s2", "c1": "s1"}


def test_cpwer_unbalanced_counts_unmapped_ref_as_deletions():
    refs = {"s1": "alpha beta gamma", "s2": "delta epsilon"}
    r = cpwer(refs, {"c0": "alpha beta gamma"})
    assert r.counts.deletions >= 2 and r.mapping["c0"] == "s1"


def test_cpwer_unmapped_ref_deletions_count_normalized_tokens():
    """Pins the consistency semantics explicitly: an unmapped speaker's
    deletions are the normalized (not raw) token count, so the pooled
    denominator stays in the same unit as every mapped pair. Computed via
    ``normalize`` itself so this survives normalizer version changes.
    """
    refs = {"s1": "one two three", "s2": "four five"}
    r = cpwer(refs, {"c0": "one two three"})
    expected_deletions = len(normalize("four five").split())
    assert r.counts.deletions == expected_deletions


def test_cpwer_unmapped_cluster_counts_as_insertions():
    refs = {"s1": "one two three"}
    hyps = {"c0": "one two three", "c1": "extra words here"}
    r = cpwer(refs, hyps)
    assert r.mapping["c0"] == "s1"
    assert r.mapping["c1"] is None
    assert r.counts.insertions >= 3
    assert r.counts.errors == 3


def test_cpwer_result_is_typed():
    r = cpwer({"s1": "hi"}, {"c0": "hi"})
    assert isinstance(r, CpWerResult)
    assert isinstance(r.counts, ErrorCounts)


def test_cpwer_empty_inputs_yield_zero_everything():
    r = cpwer({}, {})
    assert r.mapping == {}
    assert r.counts == ErrorCounts(hits=0, substitutions=0, deletions=0, insertions=0)
