"""Shared loading + summary-aggregation helpers for the measure-stage metric
battery (``asr.py``, ``speaker.py``, ``quality.py``).

Every metric module reduces per-window scalars into a run summary the same
way: mean a list of values, optionally skipping ``None`` entries, then turn
a per-key summary value that never had any data into an explicit ``None``
rather than a fabricated number. These helpers used to be byte-for-byte
copy-pasted into every metric module; they are hoisted here instead.

:func:`summary_value` (the old ``_fallback_zero``) used to default an
undefined summary key to ``0.0``. That reads as a real, precise measurement
-- e.g. ``sim_o_mean == 0.0`` looks like "orthogonal to the prompt voice",
when the true state is simply "no window in this run produced a defined
value for this key". ``None`` is returned instead (still with a logged
warning, parameterized by the calling metric's class name so the log line
stays attributable even though it's now emitted from this shared module).
``espnet3.systems.base.metric.measure`` writes each metric's returned dict
straight to ``metrics.json`` via ``json.dump``, so ``None`` serializes as
JSON ``null``; ``local/eval_report.py``'s ``_format_cell`` already renders
``None``/``null`` as ``-`` in the condition-comparison table, so an
undefined key stays visibly "no data" all the way through the pipeline
rather than silently becoming an ordinary-looking 0.0.

:func:`load_wav` / :func:`resample_wav` (moved from the now-deleted
``segments.py``, which also carried VAD/IPU machinery this lean metric suite
no longer needs) are the shared audio-loading primitives: read a wav
(recipe audio is 24 kHz) and, when a backend needs a different rate (16 kHz
for whisper / WavLM-SV / UTMOS), resample it.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

__all__ = [
    "mean",
    "mean_skip_none",
    "summary_value",
    "load_wav",
    "resample_wav",
]


# --------------------------------------------------------------------------- #
# wav loading + resampling
# --------------------------------------------------------------------------- #
def resample_wav(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 waveform via torchaudio (already a hard dep of
    this recipe's dataset/inference pipeline; not a metrics-only import)."""
    if orig_sr == target_sr:
        return wav
    wav_t = torch.as_tensor(wav, dtype=torch.float32)
    resampled = torchaudio.functional.resample(wav_t, orig_sr, target_sr)
    return resampled.numpy()


def load_wav(path: Path, target_sr: Optional[int] = None) -> "tuple[np.ndarray, int]":
    """Load a wav as mono float32, optionally resampled to ``target_sr``.

    Recipe audio is produced mono per-channel by the infer stage, but a
    stray multi-channel file (e.g. a mixdown) is downmixed defensively
    rather than rejected.

    Args:
        path: Wav file path.
        target_sr: If given and different from the file's native rate,
            resample (e.g. 24 kHz recipe audio -> 16 kHz for a backend).

    Returns:
        ``(samples, sample_rate)``.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if target_sr is not None and target_sr != sr:
        data = resample_wav(data, sr, target_sr)
        sr = target_sr
    return data, sr


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
            function is shared across all three metric modules, so the
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
