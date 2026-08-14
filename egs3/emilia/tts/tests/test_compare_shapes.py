"""compare_shapes.compare() against synthetic shape files.

The comparison tool is real arithmetic (error percentiles, batch-boundary
counting), not glue, so it gets its own coverage against hand-computable
fixtures rather than shipping untested (spec section 9's shape-parity gate
reads these numbers directly).
"""

import pytest

from egs3.emilia.tts.local.compare_shapes import compare


def _write_shape_file(path, lengths: dict, n_mels: int = 1) -> None:
    """Write '<key> <length>,<n_mels>' lines, exactly src/shape.py's format."""
    with path.open("w", encoding="utf-8") as fh:
        for key, length in lengths.items():
            fh.write(f"{key} {length},{n_mels}\n")


def test_no_overlapping_keys_raises(tmp_path):
    a = tmp_path / "a"
    c = tmp_path / "c"
    _write_shape_file(a, {"0": 10, "1": 20})
    _write_shape_file(c, {"2": 10, "3": 20})
    with pytest.raises(RuntimeError, match="No overlapping keys"):
        compare(str(a), str(c))


def test_identical_files_have_no_error_and_no_boundary_change(tmp_path):
    lengths = {str(i): 10 + i * 5 for i in range(10)}
    a = tmp_path / "a"
    c = tmp_path / "c"
    _write_shape_file(a, lengths)
    _write_shape_file(c, lengths)

    result = compare(str(a), str(c), batch_bins=105, min_batch_size=1)

    assert result["n"] == 10
    assert result["n_mismatched"] == 0
    assert result["min_err"] == 0
    assert result["max_err"] == 0
    assert result["mean_err"] == 0.0
    assert result["p50"] == result["p95"] == result["p99"] == 0.0
    assert result["n_batch_boundaries_changed"] == 0
    assert result["n_batches_analytic"] == result["n_batches_collected"]


def test_error_statistics_mostly_off_by_one_with_outliers(tmp_path):
    """100 keys: 98 with a +1 analytic/collected offset, 2 outliers at -5.

    Chosen so every returned statistic has a value computable by hand:
    err = analytic - collected, so an outlier where collected is 5 frames
    longer than analytic gives err = -5.
      abs(err) sorted: 98 ones, then two 5s (positions 0..97 and 98..99).
      numpy's default (linear) percentile interpolation then gives exact,
      not-merely-approximate values for p50/p95/p99 below.
    """
    a = tmp_path / "a"
    c = tmp_path / "c"
    analytic = {}
    collected = {}
    for i in range(100):
        length = 100 + i
        analytic[str(i)] = length
        if i < 98:
            collected[str(i)] = length - 1  # err = a - c = +1
        else:
            collected[str(i)] = length + 5  # err = a - c = -5
    _write_shape_file(a, analytic)
    _write_shape_file(c, collected)

    # batch_bins large enough that every key falls in one batch on both
    # sides, so this test isolates the error statistics from batching.
    result = compare(str(a), str(c), batch_bins=10**9, min_batch_size=1)

    assert result["n"] == 100
    assert result["n_mismatched"] == 100
    assert result["min_err"] == -5
    assert result["max_err"] == 1
    assert result["mean_err"] == pytest.approx((98 * 1 + 2 * -5) / 100)
    assert result["p50"] == pytest.approx(1.0)
    assert result["p95"] == pytest.approx(1.0)
    assert result["p99"] == pytest.approx(5.0)
    assert result["n_batch_boundaries_changed"] == 0


def test_batch_boundary_change_is_detected(tmp_path):
    """One perturbed length moves a batch-closing boundary.

    10 keys, lengths 10..100 in steps of 10 (feat_dim=1), batch_bins=105,
    min_batch_size=1. NumElementsBatchSampler packs ascending-sorted keys,
    closing a batch once `len(batch) * current_key_length > batch_bins`:

      analytic:  [10,20,30,40 -> closes at 4*40=160>105] [50,60 -> 2*60=120>105]
                 [70,80 -> 2*80=160>105] [90,100 -> 2*100=200>105]
                 => batches (0,1,2,3) (4,5) (6,7) (8,9)

    Bumping key "2" from 30 to 36 in the collected file moves its batch's
    close point earlier (3*36=108>105, closes at size 3 instead of 4), which
    cascades into the next boundary too:

      collected: [10,20,36 -> 3*36=108>105] [40,50,60 -> 3*60=180>105]
                 [70,80] [90,100]
                 => batches (0,1,2) (3,4,5) (6,7) (8,9)

    So batches 0 and 1 differ (2 changed), batches 2 and 3 are unchanged.
    """
    a = tmp_path / "a"
    c = tmp_path / "c"
    analytic = {str(i): 10 * (i + 1) for i in range(10)}
    collected = dict(analytic)
    collected["2"] = 36

    _write_shape_file(a, analytic)
    _write_shape_file(c, collected)

    result = compare(str(a), str(c), batch_bins=105, min_batch_size=1)

    assert result["n"] == 10
    assert result["n_mismatched"] == 1
    assert result["min_err"] == -6  # key "2": 30 - 36
    assert result["max_err"] == 0
    assert result["n_batches_analytic"] == 4
    assert result["n_batches_collected"] == 4
    assert result["n_batch_boundaries_changed"] == 2
