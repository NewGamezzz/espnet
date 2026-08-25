#!/usr/bin/env python3
"""Run a metrics config over an existing inference dir and MERGE the result
into its ``metrics.json`` instead of overwriting it.

``espnet3.systems.base.metric.measure`` rewrites ``metrics.json`` with only
the metrics in the config it was given, so adding one metric (e.g. the
Talking Turns judge) to an already-scored run would erase the other four.
This wrapper runs ``measure`` with the config's ``inference_dir`` redirected
to a scratch view of the same files, then merges the new class-path keys
into the real ``metrics.json`` (previous file kept as ``metrics.json.bak``).

Usage::

    python local/measure_merge.py --metrics_config conf/generated/x.yaml \
        [--only TurnTakingJudgeMetric]

The config must carry ``inference_dir`` and ``dataset.test`` like any
generated metrics config. ``--only`` keeps just the blocks whose ``_target_``
ends with the given class name.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


def merge_metrics(inference_dir: Path, new_results: dict) -> dict:
    """Merge ``new_results`` (class path -> {test: summary}) into
    ``inference_dir/metrics.json``; keep a ``.bak`` of the previous file."""
    path = inference_dir / "metrics.json"
    merged: dict = {}
    if path.exists():
        merged = json.loads(path.read_text("utf-8"))
        shutil.copy2(path, path.with_suffix(".json.bak"))
    for key, per_test in new_results.items():
        merged.setdefault(key, {}).update(per_test)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), "utf-8")
    return merged


def run(metrics_config: Path, only: str | None) -> dict:
    from espnet3.systems.base.metric import measure

    cfg = OmegaConf.load(metrics_config)
    if only:
        cfg.metrics = [m for m in cfg.metrics if m.metric._target_.endswith(only)]
        if not cfg.metrics:
            raise SystemExit(f"no metric block ends with {only!r}")
    inference_dir = Path(cfg.inference_dir)
    before = (inference_dir / "metrics.json").read_text("utf-8") if (
        inference_dir / "metrics.json"
    ).exists() else None
    results = measure(cfg)  # writes metrics.json with ONLY these metrics
    if before is not None:
        (inference_dir / "metrics.json").write_text(before, "utf-8")  # restore
    return merge_metrics(inference_dir, results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics_config", type=Path, required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    merged = run(args.metrics_config, args.only)
    print(json.dumps({k: list(v) for k, v in merged.items()}, indent=2))


if __name__ == "__main__":
    main()
