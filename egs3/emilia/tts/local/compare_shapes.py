#!/usr/bin/env python3
"""Compare analytic feats_shape against collect_stats' measured shapes.

Byte-identity is NOT the criterion. Two sources of legitimate difference:
  * centered mel framing is 1 + n_samples // hop, and n_samples comes from
    the decoder, not from the JSON duration;
  * mp3 encoder delay/padding shifts the decoded sample count.
A one or two frame error on a ~500 frame utterance is noise against a soft
batch_bins bound. A systematic offset or a long tail is the real finding.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler


def _lengths(path: str) -> dict:
    return {
        k: v[0] for k, v in load_num_sequence_text(path, loader_type="csv_int").items()
    }


def compare(
    analytic_path: str,
    collected_path: str,
    batch_bins: int = 480000,
    min_batch_size: int = 8,
) -> dict:
    a = _lengths(analytic_path)
    c = _lengths(collected_path)
    keys = sorted(set(a) & set(c), key=int)
    if not keys:
        raise RuntimeError("No overlapping keys between the two shape files")
    err = np.array([a[k] - c[k] for k in keys], dtype=np.int64)

    def batches(path):
        # Deliberately the stock sampler, not NumElementsArraySampler: this
        # isolates the shape-file difference from the max_samples cap (Task
        # 12 already pins array-vs-stock equivalence on its own), and runs
        # with the cap disabled so it pins the packing rule exactly.
        sampler = NumElementsBatchSampler(
            batch_bins=batch_bins,
            shape_files=[path],
            min_batch_size=min_batch_size,
        )
        return [tuple(sorted(b, key=int)) for b in sampler]

    ba, bc = batches(analytic_path), batches(collected_path)
    changed = sum(1 for x, y in zip(ba, bc) if x != y) + abs(len(ba) - len(bc))

    return {
        "n": len(keys),
        "n_mismatched": int((err != 0).sum()),
        "min_err": int(err.min()),
        "max_err": int(err.max()),
        "mean_err": float(err.mean()),
        "p50": float(np.percentile(np.abs(err), 50)),
        "p95": float(np.percentile(np.abs(err), 95)),
        "p99": float(np.percentile(np.abs(err), 99)),
        "n_batches_analytic": len(ba),
        "n_batches_collected": len(bc),
        "n_batch_boundaries_changed": int(changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analytic")
    parser.add_argument("collected")
    parser.add_argument("--batch_bins", type=int, default=480000)
    parser.add_argument("--min_batch_size", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.analytic, args.collected, args.batch_bins, args.min_batch_size
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
