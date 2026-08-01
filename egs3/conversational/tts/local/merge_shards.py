#!/usr/bin/env python3
"""Merge the per-shard SCPs of a sharded ``generate_external`` infer run.

A sharded run (``selection.shard_count > 1`` in the inference config) has
every shard write its own ``<name>.scp.<i>of<n>`` rather than ``<name>.scp``:
the audio and meta files are keyed by unique dialogue id and share the
directory safely, but an SCP is written wholesale and siblings would clobber
each other.  This concatenates them into the plain ``<name>.scp`` files the
``measure`` stage reads.

    python local/merge_shards.py exp/<tag>/infer_generate_external/valid

REFUSES TO WRITE A PARTIAL MERGE.  Every shard ``0..n-1`` must be present
for all five SCPs, or the run exits non-zero having written nothing.  A
silently short ``meta.scp`` would produce a metrics.json over a subset while
still looking like a complete run - exactly the failure this whole recipe
tries not to have.  Pass ``--allow-partial`` to merge what exists anyway
(it prints which shards are missing, and the result must be reported as a
partial run).

Rows are emitted in shard order, then in each shard's own order.  Metric
classes iterate ``meta.scp`` and pool their counts, so row order does not
change any summary value; it is fixed only so re-running the merge is
reproducible.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

SCP_NAMES = ("meta", "wav", "prompt", "text", "mix")
_SHARD_RE = re.compile(r"^(?P<name>[a-z]+)\.scp\.(?P<index>\d+)of(?P<count>\d+)$")


def discover(test_dir: Path) -> tuple[Dict[str, Dict[int, Path]], int]:
    """Map ``name -> {shard_index: path}`` for every per-shard SCP found,
    plus the single shard count they all agree on."""
    found: Dict[str, Dict[int, Path]] = {name: {} for name in SCP_NAMES}
    counts: set[int] = set()
    for path in sorted(test_dir.iterdir()):
        match = _SHARD_RE.match(path.name)
        if not match or match.group("name") not in found:
            continue
        found[match.group("name")][int(match.group("index"))] = path
        counts.add(int(match.group("count")))

    if not counts:
        raise SystemExit(
            f"{test_dir}: no per-shard SCPs (<name>.scp.<i>of<n>) found. "
            "An unsharded run already writes the plain names and needs no merge."
        )
    if len(counts) > 1:
        raise SystemExit(
            f"{test_dir}: mixed shard counts {sorted(counts)}. These outputs "
            "come from different shardings; merging them would double-count."
        )
    return found, counts.pop()


def missing(found: Dict[str, Dict[int, Path]], shard_count: int) -> List[str]:
    return [
        f"{name}.scp.{i}of{shard_count}"
        for name in SCP_NAMES
        for i in range(shard_count)
        if i not in found[name]
    ]


def merge(test_dir: Path, allow_partial: bool = False) -> Dict[str, int]:
    found, shard_count = discover(test_dir)
    gaps = missing(found, shard_count)
    if gaps:
        message = f"{test_dir}: missing {len(gaps)} shard file(s): " + ", ".join(
            gaps[:10]
        )
        if not allow_partial:
            raise SystemExit(
                message + "\nRefusing to write a partial merge: the result would "
                "look like a complete run. Re-run the missing shards, or pass "
                "--allow-partial and report the result as partial."
            )
        print("WARNING: " + message, file=sys.stderr)

    written: Dict[str, int] = {}
    for name in SCP_NAMES:
        rows: List[str] = []
        for index in range(shard_count):
            path = found[name].get(index)
            if path is None:
                continue
            rows.extend(
                line for line in path.read_text(encoding="utf-8").splitlines() if line
            )
        keys = [row.split(" ", 1)[0] for row in rows]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise SystemExit(
                f"{test_dir}/{name}.scp: {len(duplicates)} id(s) appear in more "
                f"than one shard, e.g. {sorted(duplicates)[:5]}. Shards must "
                "partition the selection; this would double-count in every metric."
            )
        (test_dir / f"{name}.scp").write_text(
            "".join(f"{row}\n" for row in rows), encoding="utf-8"
        )
        written[name] = len(rows)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "test_dir",
        type=Path,
        help="The infer stage's <inference_dir>/<test_name> directory.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Merge even if some shards are missing (result is a PARTIAL run).",
    )
    args = parser.parse_args(argv)

    written = merge(args.test_dir, allow_partial=args.allow_partial)
    for name, count in written.items():
        print(f"{name}.scp: {count} rows")


if __name__ == "__main__":
    main()
