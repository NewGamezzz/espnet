"""Per-speaker voice-attribute measurement for caption generation.

Pure measurement module: no I/O, no torch. Callers slice a speaker's turn
audio out of a window recording (see ``dataset.preprocessing.audio``) and
pass the resulting list of 16 kHz float32 mono arrays straight in here -
this module never reads a file or a manifest itself.

``measure_speaker`` produces the raw numeric measurements (median F0, F0
IQR, words/sec) plus the banded/categorical attributes derived from them.
Task 4 renders those bands into caption text via a banded vocabulary, so
this module's job stops at measurement + quantization, not text.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

# pyin's search range. 65 Hz covers low male voices; 400 Hz covers high
# female/child voices with headroom. Kept inside pyin's own default
# clamp range and matched to the conversational-speech domain of this
# recipe (adult speakers, not singing).
PYIN_FMIN_HZ = 65.0
PYIN_FMAX_HZ = 400.0

# Pitch bands on median F0 (Hz). DESIGN-CRITICAL: Task 4's caption
# templates key off these exact split points (low/medium/high) - do not
# change without updating the caption band vocabulary.
# low < 145 <= medium < 200 <= high
PITCH_LOW_MEDIUM_HZ = 145.0
PITCH_MEDIUM_HIGH_HZ = 200.0

# Variability band on F0 IQR (Hz). DESIGN-CRITICAL: same caption-template
# dependency as above.
# flat < 40 <= expressive
VARIABILITY_FLAT_EXPRESSIVE_HZ = 40.0

# Speaking-rate bands on words/sec. DESIGN-CRITICAL: same caption-template
# dependency as above.
# measured < 2.5 <= moderate < 3.5 <= brisk
RATE_MEASURED_MODERATE_WPS = 2.5
RATE_MODERATE_BRISK_WPS = 3.5

# Gender heuristic fallback split on median F0 (Hz), used only when no
# usable corpus metadata is available for the speaker.
# male < 165 <= female
GENDER_HEURISTIC_F0_HZ = 165.0


@dataclass(frozen=True)
class SpeakerAttrs:
    median_f0: float
    f0_iqr: float
    words_per_sec: float
    pitch_band: str
    variability_band: str
    rate_band: str
    gender: str
    gender_source: str


def _pitch_band(median_f0: float) -> str:
    if median_f0 < PITCH_LOW_MEDIUM_HZ:
        return "low"
    if median_f0 < PITCH_MEDIUM_HIGH_HZ:
        return "medium"
    return "high"


def _variability_band(f0_iqr: float) -> str:
    if f0_iqr < VARIABILITY_FLAT_EXPRESSIVE_HZ:
        return "flat"
    return "expressive"


def _rate_band(words_per_sec: float) -> str:
    if words_per_sec < RATE_MEASURED_MODERATE_WPS:
        return "measured"
    if words_per_sec < RATE_MODERATE_BRISK_WPS:
        return "moderate"
    return "brisk"


def resolve_gender(
    speaker_id: str, metadata: dict | None, median_f0: float
) -> tuple[str, str]:
    """Resolve (gender, gender_source) for ``speaker_id``.

    ``metadata``, when given, is expected keyed by speaker id, mapping to a
    per-speaker dict that may carry a ``"gender"`` or ``"sex"`` key (checked
    in that order) whose string value starts case-insensitively with "m" or
    "f" (e.g. "M", "male", "F", "Female"). If such a value is found, returns
    it with source ``"metadata"``.

    Any other shape - ``metadata`` is ``None``, ``speaker_id`` is absent,
    the per-speaker entry isn't a dict, neither key is present, or the
    value doesn't start with m/f - falls through to the pitch heuristic:
    ``median_f0 >= 165.0`` -> "female", else "male", source
    ``"pitch_heuristic"``. This heuristic is the documented fallback
    (decision 11 prefers source-corpus metadata; SSSD metadata
    availability is confirmed in Task 6).
    """
    if isinstance(metadata, dict):
        entry = metadata.get(speaker_id)
        if isinstance(entry, dict):
            for key in ("gender", "sex"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    lowered = value.strip().lower()
                    if lowered.startswith("m"):
                        return "male", "metadata"
                    if lowered.startswith("f"):
                        return "female", "metadata"

    gender = "female" if median_f0 >= GENDER_HEURISTIC_F0_HZ else "male"
    return gender, "pitch_heuristic"


def _pooled_voiced_f0(turn_wavs: list[np.ndarray], sr: int) -> np.ndarray:
    """Voiced-frame F0 values pooled across ``turn_wavs``.

    Runs ``librosa.pyin`` on each turn wav independently and concatenates
    the resulting voiced-frame F0 arrays, rather than splicing the raw
    turn audio into one array and running pyin once. Splicing raw audio
    introduces a hard discontinuity at each turn boundary; pyin's Viterbi
    decoding smooths across frames, so a spliced boundary both drops
    voiced frames near the seam and biases the estimate. Verified
    empirically: on a two-segment 150/250 Hz synthetic fixture, the
    per-turn-then-pool approach recovers 64/64 voiced frames, while
    splicing first and running pyin once recovers only 61/64 and skews
    the pooled median toward the first segment.
    """
    voiced_chunks = []
    for wav in turn_wavs:
        if wav.size == 0:
            continue
        f0, voiced_flag, _ = librosa.pyin(
            wav, fmin=PYIN_FMIN_HZ, fmax=PYIN_FMAX_HZ, sr=sr
        )
        voiced_chunks.append(f0[voiced_flag])
    if not voiced_chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(voiced_chunks)


def measure_speaker(
    turn_wavs: list[np.ndarray],
    sr: int,
    texts: list[str],
    speaker_id: str,
    metadata: dict | None = None,
) -> SpeakerAttrs:
    """Measure banded voice attributes for one speaker from their turn audio.

    ``turn_wavs`` are the speaker's own turn-slice arrays only (a caller
    upstream of this module is responsible for cutting them out of the
    session/window audio); this function does no slicing itself.

    F0: ``librosa.pyin`` (fmin=65, fmax=400 Hz) is run per turn wav and the
    voiced-frame F0 values are pooled across turns (see
    ``_pooled_voiced_f0`` for why per-turn-then-pool, not splice-then-pyin).
    ``median_f0`` is the median of the pooled voiced F0; ``f0_iqr`` is its
    75th minus 25th percentile. If pyin finds no voiced frames anywhere
    across all turns, raises ``ValueError`` naming ``speaker_id`` rather
    than fabricating attributes from an empty measurement.

    Speaking rate: total whitespace-split word count of ``texts`` divided
    by total speech seconds, i.e. ``sum(len(wav) / sr for wav in
    turn_wavs)`` - the wav durations, not the voiced-frame duration.

    Gender: delegates to ``resolve_gender(speaker_id, metadata, median_f0)``.
    """
    voiced_f0 = _pooled_voiced_f0(turn_wavs, sr)
    if voiced_f0.size == 0:
        raise ValueError(
            f"no voiced F0 frames found for speaker {speaker_id!r} across "
            f"{len(turn_wavs)} turn wav(s); cannot measure pitch attributes"
        )

    median_f0 = float(np.median(voiced_f0))
    f0_iqr = float(np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25))

    total_words = sum(len(text.split()) for text in texts)
    total_seconds = sum(len(wav) / sr for wav in turn_wavs)
    words_per_sec = total_words / total_seconds if total_seconds > 0 else 0.0

    gender, gender_source = resolve_gender(speaker_id, metadata, median_f0)

    return SpeakerAttrs(
        median_f0=median_f0,
        f0_iqr=f0_iqr,
        words_per_sec=words_per_sec,
        pitch_band=_pitch_band(median_f0),
        variability_band=_variability_band(f0_iqr),
        rate_band=_rate_band(words_per_sec),
        gender=gender,
        gender_source=gender_source,
    )
