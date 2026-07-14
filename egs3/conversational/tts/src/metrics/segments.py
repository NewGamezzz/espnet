"""Shared segment/IPU utilities for the measure-stage metric battery.

Every metric (``ConversationASRMetric``, ``SpeakerDynamicsMetric``,
``InteractionMetric``, ``ChannelQualityMetric``; the concrete classes land in
later tasks) needs the same three primitives on a per-channel wav:

* :func:`load_wav` -- read a wav (recipe audio is 24 kHz) and, when the
  caller's backend needs a different rate (16 kHz for whisper / WavLM-SV /
  UTMOS), resample it.
* :class:`VAD` -- run voice-activity detection through an injectable
  backend. The real default, :class:`SileroVADBackend`, downloads/loads its
  model lazily on first call so importing this module (or anything that
  imports it, including at collection time in CPU-only tests) never touches
  the network. Tests inject a deterministic fake/energy backend instead.
* :func:`build_ipus` -- group raw VAD speech segments into Inter-Pausal
  Units per the dGSLM rule: speech regions separated by more than
  ``min_silence`` (200 ms default) are distinct IPUs; shorter gaps are
  absorbed into one IPU. ``min_speech`` drops IPUs that are still too short
  after merging, and ``pad`` widens edges (e.g. for ASR context), re-merging
  any IPUs the padding brings back into contact.

:func:`merge_intervals`, :func:`subtract_intervals`, and :func:`solo_regions`
are re-exported (not reimplemented) from ``local/crosstalk_report.py``, which
already has tested interval algebra for the generated-bleed metric.

Nothing here is imported by the training path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from egs3.conversational.tts.local.crosstalk_report import (
    Interval,
    merge_intervals,
    solo_regions,
    subtract_intervals,
)

__all__ = [
    "Interval",
    "merge_intervals",
    "subtract_intervals",
    "solo_regions",
    "load_wav",
    "resample_wav",
    "VADBackend",
    "SileroVADBackend",
    "VAD",
    "build_ipus",
]

_EPS = 1e-9

# A VAD backend is any callable ``(wav, sr) -> Sequence[(start_sec, end_sec)]``.
VADBackend = Callable[[np.ndarray, int], Sequence[Interval]]


# --------------------------------------------------------------------------- #
# wav loading + resampling
# --------------------------------------------------------------------------- #
def resample_wav(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 waveform via torchaudio (already a hard dep of
    this recipe's dataset/inference pipeline; not a metrics-only import)."""
    if orig_sr == target_sr:
        return wav
    import torch
    import torchaudio

    wav_t = torch.as_tensor(wav, dtype=torch.float32)
    resampled = torchaudio.functional.resample(wav_t, orig_sr, target_sr)
    return resampled.numpy()


def load_wav(path: Path, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Load a wav as mono float32, optionally resampled to ``target_sr``.

    Recipe audio is produced mono per-channel by the infer stage, but a
    stray multi-channel file is downmixed defensively rather than rejected.

    Args:
        path: Wav file path.
        target_sr: If given and different from the file's native rate,
            resample (e.g. 24 kHz recipe audio -> 16 kHz for a backend).

    Returns:
        ``(samples, sample_rate)``.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if target_sr is not None and target_sr != sr:
        data = resample_wav(data, sr, target_sr)
        sr = target_sr
    return data, sr


# --------------------------------------------------------------------------- #
# VAD: injectable backend, silero as the lazy real default
# --------------------------------------------------------------------------- #
class SileroVADBackend:
    """Real default VAD backend. ``torch.hub.load`` (network fetch + model
    build) is deferred to the first call, never module import or
    ``__init__``, so constructing this class is always safe offline."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        if sample_rate not in (8000, 16000):
            raise ValueError("silero-vad only supports 8000/16000 Hz")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._get_speech_timestamps = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # local: this is the network-touching load, kept lazy

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]

    def __call__(self, wav: np.ndarray, sr: int) -> list[Interval]:
        if sr != self.sample_rate:
            raise ValueError(
                f"SileroVADBackend expects {self.sample_rate} Hz audio, got {sr}"
            )
        self._load()
        import torch

        wav_t = torch.as_tensor(wav, dtype=torch.float32)
        timestamps = self._get_speech_timestamps(
            wav_t, self._model, threshold=self.threshold, sampling_rate=sr
        )
        return [(t["start"] / sr, t["end"] / sr) for t in timestamps]


class VAD:
    """Injectable VAD wrapper around a backend callable.

    Defaults to :class:`SileroVADBackend`. Backends may be any
    ``(wav, sr) -> Sequence[(start_sec, end_sec)]`` callable, which is all
    tests need to fake VAD deterministically.
    """

    def __init__(self, backend: Optional[VADBackend] = None):
        self.backend = backend if backend is not None else SileroVADBackend()

    def __call__(self, wav: np.ndarray, sr: int) -> list[Interval]:
        raw = self.backend(wav, sr)
        return [(float(start), float(end)) for start, end in raw]


# --------------------------------------------------------------------------- #
# IPU construction: the dGSLM 200ms rule
# --------------------------------------------------------------------------- #
def build_ipus(
    speech_segments: Sequence[Interval],
    *,
    min_silence: float = 0.2,
    min_speech: float = 0.0,
    pad: float = 0.0,
    total_duration: Optional[float] = None,
) -> list[Interval]:
    """Group raw VAD speech segments into Inter-Pausal Units (dGSLM rule).

    Args:
        speech_segments: Raw speech intervals from a VAD backend (need not
            be pre-sorted or non-overlapping).
        min_silence: Gaps of AT MOST this many seconds are absorbed into one
            IPU; gaps strictly greater than this keep segments distinct.
            dGSLM default: 0.2 (200 ms).
        min_speech: IPUs shorter than this (after merging) are dropped as
            noise/blips.
        pad: Seconds to extend every IPU's edges by (e.g. so a downstream
            ASR call gets a little context). Padding that brings two IPUs
            back into contact re-merges them, so the output never contains
            overlapping or unordered intervals.
        total_duration: When given, clamps the padded upper edge (and the
            lower edge, always clamped to 0) to this bound.

    Returns:
        Sorted, non-overlapping ``(start_sec, end_sec)`` IPU intervals.
    """
    merged = merge_intervals(list(speech_segments))

    ipus: list[Interval] = []
    for start, end in merged:
        if ipus and start - ipus[-1][1] <= min_silence + _EPS:
            ipus[-1] = (ipus[-1][0], max(ipus[-1][1], end))
        else:
            ipus.append((start, end))

    ipus = [(s, e) for s, e in ipus if e - s >= min_speech - _EPS]

    if pad and ipus:
        padded = []
        for s, e in ipus:
            s2 = max(0.0, s - pad)
            e2 = e + pad
            if total_duration is not None:
                e2 = min(total_duration, e2)
            padded.append((s2, e2))
        ipus = merge_intervals(padded)

    return ipus
