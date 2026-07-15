"""Shared summary-aggregation helpers for the measure-stage metric battery.

Every metric module (``asr.py``, ``speaker.py``, ``interaction.py``,
``quality.py``) reduces per-window scalars into a run summary the same way:
mean a list of values, optionally skipping ``None`` entries, then turn a
per-key summary value that never had any data into an explicit ``None``
rather than a fabricated number. These three helpers used to be
byte-for-byte copy-pasted into all four modules (interaction.py even carried
a comment claiming this duplication was deliberate "house convention" -- it
was not; it was drift risk with no offsetting benefit). They are hoisted
here instead.

:func:`summary_value` (the old ``_fallback_zero``) used to default an
undefined summary key to ``0.0``. That reads as a real, precise measurement
-- ``bleed_db_p50 == 0.0`` looks like "generated bleed as loud as the
speaker itself" and ``kendall_tau == 0.0`` looks like "no rank correlation",
when the true state is simply "no window in this run produced a defined
value for this key". ``None`` is now returned instead (still with a logged
warning, parameterized by the calling metric's class name so the log line
stays attributable even though it's now emitted from this shared module).
``espnet3.systems.base.metric.measure`` writes each metric's returned dict
straight to ``metrics.json`` via ``json.dump``, so ``None`` serializes as
JSON ``null``; ``local/eval_report.py``'s ``_format_cell`` already renders
``None``/``null`` as ``-`` in the condition-comparison table, so an
undefined key stays visibly "no data" all the way through the pipeline
rather than silently becoming an ordinary-looking 0.0.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["mean", "mean_skip_none", "summary_value"]


def mean(values: Sequence[float]) -> Optional[float]:
    """Plain mean; ``None`` for an empty sequence (never a fabricated 0)."""
    values = list(values)
    return sum(values) / len(values) if values else None


def mean_skip_none(values) -> Optional[float]:
    """Mean over the non-``None`` entries of ``values``; ``None`` if none
    are defined."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summary_value(
    value: Optional[float], key: str, *, metric_name: str
) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` (with a logged warning) when
    no window in the run produced a defined value for ``key``.

    Args:
        value: The aggregated (e.g. mean-of-windows) value for this summary
            key, or ``None`` if nothing contributed to it.
        key: The summary key's name, for the warning message.
        metric_name: The calling metric class's name (e.g.
            ``"ConversationASRMetric"``), for the warning message -- this
            function is shared across all four metric modules, so the
            caller must identify itself explicitly.
    """
    if value is None:
        logger.warning(
            "%s: no window produced a defined value for '%s'; leaving the "
            "run summary undefined (serializes as null in metrics.json)",
            metric_name,
            key,
        )
        return None
    return float(value)
