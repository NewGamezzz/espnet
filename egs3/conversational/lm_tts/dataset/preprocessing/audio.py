"""Audio tail: window slicing, resampling, and channel/mix wav writing.

Pure module: ``soundfile`` + ``soxr`` only, no torch. Cuts a ``WindowRecord``
span out of its source multi-channel session recording and writes 16 kHz
mono PCM16 wavs: one per channel plus a mixed-mono file.

DESIGN-CRITICAL invariant (decision 12): there is exactly one resample path
in this recipe, ``load_window_channel``, and every other function builds on
its output. ``mix_mono`` and ``cut_window_wavs`` never re-resample or
re-read a channel to build the mix - the mix is always the sum of the exact
float32 arrays that get quantized into the per-channel wav files. This
keeps the TAC per-channel wavs and the mono-baseline mix sample-consistent
with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .windows import WindowRecord


def load_window_channel(
    audio_path: str | Path,
    t0: float,
    t1: float,
    channel: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Read one channel of ``[t0, t1)`` from ``audio_path``, resampled to ``target_sr``.

    Reads only the requested slice via ``soundfile.read(..., start=, stop=)``
    (never loads the whole session file) and resamples with ``soxr`` at
    ``HQ`` quality - the single resample path shared by every caller in this
    module, including the mix built by ``mix_mono``.
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    start = int(round(t0 * sr))
    stop = min(int(round(t1 * sr)), info.frames)
    data, read_sr = sf.read(
        str(audio_path), start=start, stop=stop, dtype="float32", always_2d=True
    )
    if channel < 0 or channel >= data.shape[1]:
        raise ValueError(
            f"channel {channel} out of range for {audio_path} with "
            f"{data.shape[1]} channels"
        )
    mono = np.ascontiguousarray(data[:, channel], dtype=np.float32)
    if read_sr != target_sr:
        mono = soxr.resample(mono, read_sr, target_sr, quality="HQ").astype(np.float32)
    return mono


def mix_mono(channels: list[np.ndarray]) -> np.ndarray:
    """Sum ``channels`` into one mono array with clip-guard scaling.

    Scales the raw sum by ``max(1.0, peak)`` where ``peak`` is the max abs
    sample of that sum, so quiet mixes pass through untouched and only
    mixes that would clip get pulled back inside [-1, 1].
    """
    if not channels:
        raise ValueError("mix_mono requires at least one channel")
    length = len(channels[0])
    for c in channels:
        if len(c) != length:
            raise ValueError("all channels must have the same sample count")
    total = np.zeros(length, dtype=np.float32)
    for c in channels:
        total = total + c
    peak = float(np.max(np.abs(total))) if length else 0.0
    scale = max(1.0, peak)
    return (total / scale).astype(np.float32)


@dataclass(frozen=True)
class WindowAudio:
    window_id: str
    channel_paths: tuple[Path, ...]
    mix_path: Path
    channel_durations: tuple[float, ...]
    mix_duration: float


def cut_window_wavs(
    record: WindowRecord,
    dataset_root: str | Path,
    out_dir: str | Path,
    target_sr: int = 16000,
) -> WindowAudio:
    """Cut ``record``'s window out of its source recording into wav files.

    Writes ``<window_id>_ch{i}.wav`` for each channel (``i < num_channels``)
    and ``<window_id>_mix.wav`` under ``out_dir``, all 16 kHz mono PCM16.
    The mix is summed from the exact float32 arrays quantized into the
    channel wavs (decision 12); it is never resampled or re-read separately.
    """
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = dataset_root / record.audio_relpath

    channels = [
        load_window_channel(audio_path, record.t0, record.t1, ch, target_sr)
        for ch in range(record.num_channels)
    ]

    channel_paths: list[Path] = []
    channel_durations: list[float] = []
    for ch, arr in enumerate(channels):
        path = out_dir / f"{record.window_id}_ch{ch}.wav"
        sf.write(str(path), arr, target_sr, subtype="PCM_16")
        channel_paths.append(path)
        channel_durations.append(len(arr) / target_sr)

    mix = mix_mono(channels)
    mix_path = out_dir / f"{record.window_id}_mix.wav"
    sf.write(str(mix_path), mix, target_sr, subtype="PCM_16")
    mix_duration = len(mix) / target_sr

    return WindowAudio(
        window_id=record.window_id,
        channel_paths=tuple(channel_paths),
        mix_path=mix_path,
        channel_durations=tuple(channel_durations),
        mix_duration=mix_duration,
    )
