"""Active-speech loudness: the -23 dBFS prompt convention, shared by the
single-call path (``src/inference.py``), the ZipVoice-Dialog v2 builder and
the AMI long-form / baseline exporters.

The level is the RMS over 20 ms frames whose RMS exceeds an *active*
threshold (linear).  The eval suite's fixed floor (``1e-3``, -60 dBFS) is
fine for studio-clean corpora, but AMI headsets idle at a median -51 dBFS
(gate measurement 2026-09-06: 70 of 96 floors above -60 dBFS), so there
the threshold must sit above the channel's own floor or noise frames count
as speech and the level is under-read.  ``threshold_from_floor`` derives
it from the prompt gate's measured floor plus a margin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ACTIVE_THRESHOLD = 1e-3  # -60 dBFS, the eval suite's ACTIVE_RMS_THRESHOLD
FRAME_SEC = 0.02
PEAK_CEILING = 0.99  # PCM_16 output must not clip


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * math.log10(max(x, 1e-12)))


def threshold_from_floor(floor_db: float | None, margin_db: float) -> float:
    """Active threshold (linear): the channel floor + ``margin_db``, never
    below the default; the default alone when no floor is known."""
    if floor_db is None:
        return DEFAULT_ACTIVE_THRESHOLD
    return max(DEFAULT_ACTIVE_THRESHOLD, db_to_lin(float(floor_db) + float(margin_db)))


def frame_rms(x: np.ndarray, fs: int, frame_sec: float = FRAME_SEC) -> np.ndarray:
    frame = max(1, int(fs * frame_sec))
    n = x.shape[0] // frame
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt(np.square(x[: n * frame].reshape(n, frame).astype(np.float64)).mean(axis=1))


def active_rms_db(
    x: np.ndarray, fs: int, threshold: float = DEFAULT_ACTIVE_THRESHOLD
) -> float | None:
    """Active-speech level in dBFS, ``None`` when no frame is active."""
    rms = frame_rms(x, fs)
    rms = rms[rms > threshold]
    if not len(rms):
        return None
    return lin_to_db(float(np.sqrt(np.square(rms).mean())))


def gain_to_target(
    level_source: np.ndarray,
    fs: int,
    target_db: float,
    *,
    threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    peak: float | None = None,
) -> tuple[float, bool, float | None]:
    """Linear gain that moves ``level_source``'s active RMS to ``target_db``.

    Returns ``(gain, peak_limited, level_db)``.  ``peak`` is the absolute
    peak of everything the gain will be applied to (defaults to the level
    source's own); the gain is capped so that peak stays <= 0.99 and the
    cap is reported, never silently applied.  No active frame -> gain 1.0.
    """
    level = active_rms_db(level_source, fs, threshold)
    if level is None:
        return 1.0, False, None
    gain = db_to_lin(float(target_db) - level)
    if peak is None:
        peak = float(np.abs(level_source).max()) if level_source.size else 0.0
    limited = peak * gain > PEAK_CEILING
    if limited:
        gain = PEAK_CEILING / peak
    return float(gain), bool(limited), float(level)


def load_channel_floors(path) -> dict[str, list[float | None]]:
    """``floor_db`` from the prompt gate's ``exclude_spans.json`` (version 1):
    per session, one idle-level (dBFS) per SOURCE channel.  ``{}`` when the
    file carries none (older sidecars)."""
    data = json.loads(Path(path).read_text("utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"{path}: unsupported exclude_spans version {data.get('version')!r}")
    out: dict[str, list[float | None]] = {}
    for sid, floors in (data.get("floor_db") or {}).items():
        out[str(sid)] = [None if f is None else float(f) for f in floors]
    return out


def loudness_meta(
    target_db: float | None,
    threshold: list[float] | None,
    gains: list[float] | None,
    limited: list[bool] | None,
    levels: list[float | None] | None,
) -> dict[str, Any] | None:
    """The per-channel record written to meta (``None`` when the knob is off)."""
    if target_db is None:
        return None
    return {
        "target_db": float(target_db),
        "threshold_db": [round(lin_to_db(t), 3) for t in (threshold or [])],
        "level_db": [None if v is None else round(v, 3) for v in (levels or [])],
        "gain_db": [round(lin_to_db(g), 3) for g in (gains or [])],
        "peak_limited": [bool(b) for b in (limited or [])],
    }
