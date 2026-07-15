"""Tests for per-speaker voice-attribute measurement (pitch/rate/gender bands)."""

import numpy as np
import pytest

from dataset.preprocessing.attributes import (
    PITCH_LOW_MEDIUM_HZ,
    PITCH_MEDIUM_HIGH_HZ,
    RATE_MEASURED_MODERATE_WPS,
    RATE_MODERATE_BRISK_WPS,
    VARIABILITY_FLAT_EXPRESSIVE_HZ,
    SpeakerAttrs,
    measure_speaker,
    resolve_gender,
)

SR = 16000


def tone(freq, dur=1.0, sr=SR, amp=0.3):
    """A pure sine tone at ``freq`` Hz, ``dur`` seconds, float32."""
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMeasureSpeakerPitchBands:
    """median_f0 quantized into low/medium/high; synthetic tones with known F0."""

    def test_low_pitch_tone_lands_in_low_band(self):
        attrs = measure_speaker([tone(120.0)], SR, ["hello there"], "spk_low")
        assert attrs.pitch_band == "low"
        assert attrs.median_f0 == pytest.approx(120.0, abs=5.0)

    def test_medium_pitch_tone_lands_in_medium_band(self):
        attrs = measure_speaker([tone(180.0)], SR, ["hello there"], "spk_mid")
        assert attrs.pitch_band == "medium"
        assert attrs.median_f0 == pytest.approx(180.0, abs=5.0)

    def test_high_pitch_tone_lands_in_high_band(self):
        attrs = measure_speaker([tone(220.0)], SR, ["hello there"], "spk_high")
        assert attrs.pitch_band == "high"
        assert attrs.median_f0 == pytest.approx(220.0, abs=5.0)

    def test_pitch_band_boundary_constants_match_brief(self):
        # Caption templates depend on these exact numbers (brief-mandated).
        assert PITCH_LOW_MEDIUM_HZ == 145.0
        assert PITCH_MEDIUM_HIGH_HZ == 200.0


class TestMeasureSpeakerVariabilityBand:
    """F0 IQR quantized into flat/expressive."""

    def test_steady_tone_is_flat(self):
        attrs = measure_speaker([tone(150.0)], SR, ["hello there"], "spk_steady")
        assert attrs.variability_band == "flat"
        assert attrs.f0_iqr < VARIABILITY_FLAT_EXPRESSIVE_HZ

    def test_alternating_pitch_segments_are_expressive(self):
        """Two turn wavs at 150 Hz and 250 Hz: pooling voiced F0 across turns
        spans a 100 Hz gap, well past the 40 Hz flat/expressive split."""
        attrs = measure_speaker(
            [tone(150.0), tone(250.0)],
            SR,
            ["hello there", "how are you"],
            "spk_expressive",
        )
        assert attrs.variability_band == "expressive"
        assert attrs.f0_iqr >= VARIABILITY_FLAT_EXPRESSIVE_HZ

    def test_variability_boundary_constant_matches_brief(self):
        assert VARIABILITY_FLAT_EXPRESSIVE_HZ == 40.0


class TestMeasureSpeakerRateBands:
    """words_per_sec = total words / total speech seconds, quantized into
    measured/moderate/brisk with >= at each boundary."""

    def test_slow_rate_is_measured(self):
        # 1 word/sec worth of text over a 1s tone -> well under 2.5.
        attrs = measure_speaker([tone(150.0, dur=2.0)], SR, ["one two"], "spk_slow")
        assert attrs.rate_band == "measured"
        assert attrs.words_per_sec == pytest.approx(1.0, abs=0.01)

    def test_moderate_rate_boundary_is_inclusive(self):
        # 5 words over 2.0s = 2.5 words/sec exactly -> moderate (>= convention).
        attrs = measure_speaker(
            [tone(150.0, dur=2.0)], SR, ["one two three four five"], "spk_exact25"
        )
        assert attrs.words_per_sec == pytest.approx(2.5, abs=1e-6)
        assert attrs.rate_band == "moderate"

    def test_brisk_rate_boundary_is_inclusive(self):
        # 7 words over 2.0s = 3.5 words/sec exactly -> brisk (>= convention).
        attrs = measure_speaker(
            [tone(150.0, dur=2.0)],
            SR,
            ["one two three four five six seven"],
            "spk_exact35",
        )
        assert attrs.words_per_sec == pytest.approx(3.5, abs=1e-6)
        assert attrs.rate_band == "brisk"

    def test_just_below_moderate_boundary_is_measured(self):
        # 4 words over 2.0s = 2.0 words/sec -> measured.
        attrs = measure_speaker(
            [tone(150.0, dur=2.0)], SR, ["one two three four"], "spk_below25"
        )
        assert attrs.rate_band == "measured"

    def test_just_below_brisk_boundary_is_moderate(self):
        # 6 words over 2.0s = 3.0 words/sec -> moderate.
        attrs = measure_speaker(
            [tone(150.0, dur=2.0)], SR, ["one two three four five six"], "spk_below35"
        )
        assert attrs.rate_band == "moderate"

    def test_rate_sums_words_and_durations_across_turns(self):
        # 3 turns: 3+2+1 = 6 words over 1.0+1.0+1.0 = 3.0s -> 2.0 words/sec.
        attrs = measure_speaker(
            [tone(150.0, dur=1.0), tone(150.0, dur=1.0), tone(150.0, dur=1.0)],
            SR,
            ["one two three", "four five", "six"],
            "spk_multiturn",
        )
        assert attrs.words_per_sec == pytest.approx(2.0, abs=0.01)

    def test_rate_boundary_constants_match_brief(self):
        assert RATE_MEASURED_MODERATE_WPS == 2.5
        assert RATE_MODERATE_BRISK_WPS == 3.5


class TestMeasureSpeakerNoVoicedFrames:
    def test_silence_raises_value_error_naming_speaker(self):
        silence = np.zeros(SR, dtype=np.float32)
        with pytest.raises(ValueError, match="spk_silent"):
            measure_speaker([silence], SR, ["hello"], "spk_silent")

    def test_white_noise_raises_value_error(self):
        rng = np.random.default_rng(0)
        noise = (0.3 * rng.standard_normal(SR)).astype(np.float32)
        with pytest.raises(ValueError, match="spk_noisy"):
            measure_speaker([noise], SR, ["hello"], "spk_noisy")


class TestMeasureSpeakerReturnType:
    def test_returns_frozen_speaker_attrs_dataclass(self):
        attrs = measure_speaker([tone(150.0)], SR, ["hello there"], "spk0")
        assert isinstance(attrs, SpeakerAttrs)
        with pytest.raises(Exception):
            attrs.median_f0 = 999.0  # frozen dataclass must reject mutation


class TestResolveGender:
    """metadata (keyed by speaker_id, "gender"/"sex" keys) beats the pitch
    heuristic (>=165 Hz female, else male)."""

    def test_metadata_gender_key_wins_over_heuristic(self):
        # Low F0 would heuristically say male; metadata says female.
        gender, source = resolve_gender(
            "spk0", {"spk0": {"gender": "F"}}, median_f0=100.0
        )
        assert gender == "female"
        assert source == "metadata"

    def test_metadata_sex_key_accepted(self):
        gender, source = resolve_gender(
            "spk0", {"spk0": {"sex": "male"}}, median_f0=220.0
        )
        assert gender == "male"
        assert source == "metadata"

    def test_metadata_value_case_insensitive_full_word(self):
        gender, source = resolve_gender(
            "spk0", {"spk0": {"gender": "Male"}}, median_f0=220.0
        )
        assert gender == "male"
        assert source == "metadata"

    def test_metadata_missing_speaker_falls_through_to_heuristic(self):
        gender, source = resolve_gender("spk_unknown", {"spk0": {"gender": "F"}}, 100.0)
        assert gender == "male"
        assert source == "pitch_heuristic"

    def test_metadata_none_falls_through_to_heuristic(self):
        gender, source = resolve_gender("spk0", None, median_f0=100.0)
        assert gender == "male"
        assert source == "pitch_heuristic"

    def test_metadata_garbage_value_falls_through_to_heuristic(self):
        gender, source = resolve_gender(
            "spk0", {"spk0": {"gender": "unknown"}}, median_f0=100.0
        )
        assert gender == "male"
        assert source == "pitch_heuristic"

    def test_metadata_non_dict_speaker_entry_falls_through_to_heuristic(self):
        gender, source = resolve_gender("spk0", {"spk0": "female"}, median_f0=100.0)
        assert gender == "male"
        assert source == "pitch_heuristic"

    def test_heuristic_high_f0_is_female(self):
        gender, source = resolve_gender("spk0", None, median_f0=220.0)
        assert gender == "female"
        assert source == "pitch_heuristic"

    def test_heuristic_boundary_165_is_female_inclusive(self):
        gender, source = resolve_gender("spk0", None, median_f0=165.0)
        assert gender == "female"
        assert source == "pitch_heuristic"

    def test_heuristic_just_below_boundary_is_male(self):
        gender, source = resolve_gender("spk0", None, median_f0=164.9)
        assert gender == "male"
        assert source == "pitch_heuristic"


class TestMeasureSpeakerGenderIntegration:
    def test_measure_speaker_calls_resolve_gender_with_metadata(self):
        # Low-F0 tone (heuristically male) with metadata overriding to female.
        attrs = measure_speaker(
            [tone(120.0)],
            SR,
            ["hello there"],
            "spk0",
            metadata={"spk0": {"gender": "female"}},
        )
        assert attrs.gender == "female"
        assert attrs.gender_source == "metadata"

    def test_measure_speaker_falls_back_to_heuristic_without_metadata(self):
        attrs = measure_speaker([tone(220.0)], SR, ["hello there"], "spk0")
        assert attrs.gender == "female"
        assert attrs.gender_source == "pitch_heuristic"
