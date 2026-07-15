"""Condition-comparison report: read ``metrics.json`` from several ``infer``
output directories (conditions -- e.g. ``gt`` / ``resynth`` / ``pretrained``
/ ``finetuned``) and emit ONE Markdown table per metric class, conditions as
columns, summary keys as rows.

Each ``metrics.json`` (written by ``espnet3.systems.base.metric.measure``,
see ``src/inference.py``'s and ``src/metrics/*.py``'s module docstrings for
the exact shape) is ``{class_path: {test_name: {summary_key: value}}}``.
This script does not import or instantiate anything from ``src/metrics/``;
it only reads the JSON already written to disk, so it never touches a
model/backend and never needs the eval-only dependencies (faster-whisper,
WavLM-SV, UTMOS, silero-vad, ...).

A condition missing ``metrics.json`` entirely, or missing a particular
metric class, or missing the requested ``test_name``, or missing an
individual summary key, all render as ``-`` in that cell -- this script
never raises on missing data, only on genuinely malformed JSON.

Usage (from the recipe dir):

    python local/eval_report.py \\
        --label gt exp/eval_pretrained/infer_gt \\
        --label resynth exp/eval_pretrained/infer_resynth \\
        --label pretrained exp/eval_pretrained/infer_generate \\
        -o exp/eval_pretrained/report.md

``--label NAME DIR`` may be repeated for as many conditions as needed;
columns appear in the order given.  Omit ``-o`` to print the report to
stdout instead of writing a file.  ``--test-name`` (default ``valid``)
selects which test-set entry of each metric's ``metrics.json`` to report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Declared order the four metric classes are documented in (PLAN-step4.md /
# conf/metrics.yaml); any additional/unknown metric class found in a
# metrics.json is appended afterward, alphabetically, so the report never
# silently drops a class it doesn't recognize.
_CANONICAL_ORDER = [
    "ConversationASRMetric",
    "SpeakerDynamicsMetric",
    "InteractionMetric",
    "ChannelQualityMetric",
]

Condition = Tuple[str, Dict[str, Any]]


def load_metrics_json(inference_dir: Path) -> Dict[str, Any]:
    """Load ``<inference_dir>/metrics.json``; ``{}`` if it does not exist.

    A missing file is a normal condition (that ``infer``/``measure`` run
    simply has not happened yet), not an error -- the report renders that
    whole column as ``-`` rather than crashing.
    """
    path = Path(inference_dir) / "metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _short_class_name(class_path: str) -> str:
    return class_path.rsplit(".", 1)[-1]


def build_sections(
    conditions: Sequence[Condition], test_name: str
) -> Dict[str, Dict[str, Any]]:
    """Pivot ``[(label, metrics_json), ...]`` into one section per metric
    class: ``{short_class_name: {"keys": [...], "values": {label: {key: v}}}}``.

    ``keys`` is the union of summary keys across every condition that
    defined this metric class, in first-seen order (condition order, then
    key order within each condition's summary dict) so the row order is
    deterministic and stable as new conditions are appended.  A condition
    that never produced this metric class (missing file, metric not run, or
    the requested ``test_name`` absent) is simply absent from ``values``;
    :func:`render_markdown` renders that as ``-`` for every row.
    """
    sections: Dict[str, Dict[str, Any]] = {}
    for label, metrics_json in conditions:
        for class_path, per_test in metrics_json.items():
            summary = per_test.get(test_name)
            if summary is None:
                continue
            name = _short_class_name(class_path)
            section = sections.setdefault(name, {"keys": [], "values": {}})
            for key in summary:
                if key not in section["keys"]:
                    section["keys"].append(key)
            section["values"][label] = summary
    return sections


def _section_order(names) -> List[str]:
    known = [n for n in _CANONICAL_ORDER if n in names]
    unknown = sorted(n for n in names if n not in _CANONICAL_ORDER)
    return known + unknown


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(
    labels: Sequence[str], sections: Dict[str, Dict[str, Any]], test_name: str
) -> str:
    """Render the pivoted ``sections`` into one Markdown table per metric
    class, conditions (``labels``, in the given order) as columns, summary
    keys as rows.  Cells for a condition/key combination with no value
    render as ``-``, never raising."""
    lines = [f"# Evaluation report (test set: `{test_name}`)", ""]

    if not sections:
        lines.append("No metrics found for any condition.")
        lines.append("")
        return "\n".join(lines)

    for name in _section_order(sections):
        section = sections[name]
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for key in section["keys"]:
            row = [key]
            for label in labels:
                row.append(_format_cell(section["values"].get(label, {}).get(key)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label",
        nargs=2,
        metavar=("NAME", "INFERENCE_DIR"),
        action="append",
        dest="labels",
        help=(
            "One condition column: a display name and its infer-stage "
            "output directory (the one containing metrics.json). "
            "Repeat for every condition; columns appear in the order given."
        ),
    )
    parser.add_argument(
        "--test-name",
        default="valid",
        help="Test-set entry to read from each metrics.json (default: valid).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write the report here; omit to print to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.labels:
        print("error: at least one --label NAME INFERENCE_DIR is required")
        print(
            "help: python local/eval_report.py --label gt exp/.../infer_gt "
            "--label pretrained exp/.../infer_generate [-o report.md]"
        )
        return 2

    names = [name for name, _ in args.labels]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        # Same-named conditions would silently collapse into one column,
        # dropping every earlier condition's data; fail loudly instead.
        print(f"error: duplicate --label name(s): {', '.join(duplicates)}")
        print("help: give every condition a unique NAME, e.g. --label gt_v2 <dir>")
        return 2

    conditions: List[Condition] = [
        (name, load_metrics_json(Path(inference_dir)))
        for name, inference_dir in args.labels
    ]
    labels = [name for name, _ in conditions]
    sections = build_sections(conditions, test_name=args.test_name)
    report = render_markdown(labels, sections, test_name=args.test_name)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
