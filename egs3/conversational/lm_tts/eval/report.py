"""Markdown comparison report + per-window CSV writer (Task 8).

Loads one or more run-level ``results.json`` files - each
``{"rows": [...], "aggregate": {...}}``, the exact shape ``eval.run_eval
.main`` writes - and renders one Markdown comparison table across runs,
plus a ``<name>_windows.csv`` (every row, flattened) and a
best/worst-5-by-cpWER section per run, all written alongside the report.

Reads only JSON/CSV, no audio and no models, so importing this module -
like ``eval.run_eval`` - must never pull in torch/transformers/pyannote
(enforced by the subprocess import-hygiene test in
``eval/tests/test_run_eval.py``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.metrics.wer import ErrorCounts

_TABLE_COLUMNS = (
    "WER_concat",
    "cpWER",
    "SIM_own",
    "SIM_margin",
    "sim_cross_gt",
    "UTMOS",
    "n_err",
)

# Matches eval.run_eval._ROW_FIELDS / evaluate_record's row schema exactly -
# the CSV header this module writes for every run's per-window dump.
_ROW_FIELDS = (
    "example_id",
    "n_clusters",
    "wer_concat_counts",
    "cpwer_counts",
    "cpwer_mapping",
    "mapping_disagrees",
    "sim_own_mean",
    "sim_margin_mean",
    "sim_cross_gt",
    "utmos",
    "purity_gt",
    "duration_s",
    "error",
)


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_results(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _table_row(name: str, agg: dict) -> list[str]:
    return [
        name,
        _fmt(agg["wer_concat"]["wer"]),
        _fmt(agg["cpwer"]["wer"]),
        _fmt(agg.get("sim_own_mean")),
        _fmt(agg.get("sim_margin_mean")),
        _fmt(agg.get("sim_cross_gt_mean")),
        _fmt(agg.get("utmos_mean")),
        _fmt(agg.get("n_err")),
    ]


def _row_wer(row: dict) -> float | None:
    """cpWER when the row has usable cpWER counts; falls back to
    wer_concat when cpWER is null (every Set B row, or a Set A row whose
    cpWER stage itself errored) - the brief's best/worst-5 ranking key.
    ``None`` when neither is usable (an errored row, or a zero-ref-word
    row where ``ErrorCounts.wer`` would raise).
    """
    for counts in (row.get("cpwer_counts"), row.get("wer_concat_counts")):
        if counts:
            ec = ErrorCounts(**counts)
            if ec.ref_words > 0:
                return ec.wer
    return None


def _best_worst_section(name: str, rows: list[dict]) -> list[str]:
    ranked = sorted(
        (
            (row["example_id"], _row_wer(row))
            for row in rows
            if _row_wer(row) is not None
        ),
        key=lambda pair: pair[1],
    )
    lines = [f"### {name}: best/worst 5 by cpWER (fallback WER_concat)", ""]
    best = ", ".join(f"{eid} ({wer:.4f})" for eid, wer in ranked[:5])
    worst = ", ".join(f"{eid} ({wer:.4f})" for eid, wer in ranked[-5:][::-1])
    lines.append(f"Best 5: {best or 'N/A'}")
    lines.append(f"Worst 5: {worst or 'N/A'}")
    lines.append("")
    return lines


def _write_windows_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            flat = {
                field: (
                    json.dumps(row.get(field))
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                )
                for field in _ROW_FIELDS
            }
            writer.writerow(flat)


def write_report(results_paths: dict[str, str], out_md: Path) -> None:
    """Render the cross-run comparison table plus per-run
    ``<name>_windows.csv`` and best/worst-5 sections.

    ``results_paths`` maps a run name (e.g. ``"sssd_gt_anchor"``,
    ``"sssd_vllm"``, ``"sft_vllm"``) to that run's ``results.json`` path.
    The CSVs and ``out_md`` are all written into ``out_md``'s directory.
    """
    out_md = Path(out_md)
    out_dir = out_md.parent

    lines = ["# BagPiper Eval Report", ""]
    lines.append("| run | " + " | ".join(_TABLE_COLUMNS) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |")

    best_worst_lines: list[str] = []
    for name, path in results_paths.items():
        results = _load_results(path)
        rows = results["rows"]
        agg = results["aggregate"]

        lines.append("| " + " | ".join(_table_row(name, agg)) + " |")

        _write_windows_csv(rows, out_dir / f"{name}_windows.csv")
        best_worst_lines.extend(_best_worst_section(name, rows))

    lines.append("")
    lines.extend(best_worst_lines)

    out_md.write_text("\n".join(lines), encoding="utf-8")
