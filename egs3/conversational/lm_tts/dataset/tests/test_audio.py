"""Tests for the audio tail: window slicing, resampling, channel/mix wavs."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import soxr

from dataset.preprocessing.audio import (
    WindowAudio,
    cut_window_wavs,
    load_window_channel,
    mix_mono,
)
from dataset.preprocessing.sssd import Turn
from dataset.preprocessing.windows import WindowRecord

SOURCE_SR = 48000
TARGET_SR = 16000


def _make_source(dataset_root, duration=2.0, freqs=(220.0, 440.0)):
    """Write a synthetic 2-channel 48 kHz fixture with distinct per-channel
    tones under ``dataset_root/original/sess1_mixed.wav``."""
    n = int(duration * SOURCE_SR)
    t = np.arange(n) / SOURCE_SR
    channels = [0.3 * np.sin(2 * np.pi * f * t).astype(np.float32) for f in freqs]
    data = np.stack(channels, axis=1)
    path = dataset_root / "original" / "sess1_mixed.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SOURCE_SR, subtype="PCM_16")
    return path


def _make_record(window_id="sess1_w00000", t0=0.5, t1=1.5, num_channels=2):
    return WindowRecord(
        window_id=window_id,
        session_id="sess1",
        audio_relpath="original/sess1_mixed.wav",
        num_channels=num_channels,
        sample_rate=SOURCE_SR,
        t0=t0,
        t1=t1,
        turns=(Turn(channel=0, speaker="spk0", text="hi", start=t0, end=t1),),
    )


class TestLoadWindowChannel:
    """Single resample path: partial read + one soxr resample to 16k mono."""

    def test_returns_float32_mono_at_target_sr(self, tmp_path):
        audio_path = _make_source(tmp_path)
        arr = load_window_channel(audio_path, 0.5, 1.5, channel=0, target_sr=TARGET_SR)
        assert arr.dtype == np.float32
        assert arr.ndim == 1
        assert arr.shape[0] == pytest.approx(TARGET_SR * 1.0, abs=2)

    def test_channels_have_distinct_content(self, tmp_path):
        audio_path = _make_source(tmp_path)
        ch0 = load_window_channel(audio_path, 0.5, 1.5, channel=0, target_sr=TARGET_SR)
        ch1 = load_window_channel(audio_path, 0.5, 1.5, channel=1, target_sr=TARGET_SR)
        assert not np.allclose(ch0, ch1)

    def test_partial_read_never_loads_whole_session(self, tmp_path):
        audio_path = _make_source(tmp_path, duration=10.0)
        with patch("dataset.preprocessing.audio.sf.read", wraps=sf.read) as spy:
            load_window_channel(audio_path, 5.0, 5.5, channel=0, target_sr=TARGET_SR)
        _, kwargs = spy.call_args
        assert "start" in kwargs and "stop" in kwargs
        assert kwargs["start"] == pytest.approx(5.0 * SOURCE_SR, abs=1)
        assert kwargs["stop"] - kwargs["start"] == pytest.approx(0.5 * SOURCE_SR, abs=1)

    def test_out_of_range_channel_raises(self, tmp_path):
        audio_path = _make_source(tmp_path)
        with pytest.raises(ValueError):
            load_window_channel(audio_path, 0.5, 1.5, channel=2, target_sr=TARGET_SR)


class TestMixMono:
    """Sum + clip-guard scaling by max(1.0, peak)."""

    def test_sum_without_clipping_is_unscaled(self):
        a = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        b = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        mix = mix_mono([a, b])
        np.testing.assert_allclose(mix, a + b, atol=1e-6)
        assert np.max(np.abs(mix)) <= 1.0

    def test_clip_guard_scales_down_when_peak_exceeds_one(self):
        a = np.array([0.9, -0.9], dtype=np.float32)
        b = np.array([0.9, -0.9], dtype=np.float32)
        mix = mix_mono([a, b])
        peak = float(np.max(np.abs(a + b)))
        expected = (a + b) / peak
        np.testing.assert_allclose(mix, expected, atol=1e-6)
        assert np.max(np.abs(mix)) <= 1.0 + 1e-6

    def test_single_channel_passthrough_when_below_one(self):
        a = np.array([0.1, -0.1], dtype=np.float32)
        mix = mix_mono([a])
        np.testing.assert_allclose(mix, a, atol=1e-6)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            mix_mono([np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32)])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            mix_mono([])


class TestCutWindowWavs:
    """Writes ch{0,1}/mix wavs and shares one resample path across them."""

    def test_writes_expected_files_at_16k_mono_pcm16(self, tmp_path):
        dataset_root = tmp_path / "root"
        _make_source(dataset_root)
        out_dir = tmp_path / "out"
        record = _make_record()

        result = cut_window_wavs(record, dataset_root, out_dir)

        assert isinstance(result, WindowAudio)
        assert len(result.channel_paths) == record.num_channels
        expected_names = {
            f"{record.window_id}_ch{i}.wav" for i in range(record.num_channels)
        }
        expected_names.add(f"{record.window_id}_mix.wav")
        assert {p.name for p in result.channel_paths} | {
            result.mix_path.name
        } == expected_names
        for path in list(result.channel_paths) + [result.mix_path]:
            assert path.exists()
            info = sf.info(str(path))
            assert info.samplerate == TARGET_SR
            assert info.channels == 1
            assert info.subtype == "PCM_16"

    def test_durations_match_window_length(self, tmp_path):
        dataset_root = tmp_path / "root"
        _make_source(dataset_root)
        out_dir = tmp_path / "out"
        record = _make_record(t0=0.5, t1=1.5)

        result = cut_window_wavs(record, dataset_root, out_dir)

        for d in result.channel_durations:
            assert d == pytest.approx(1.0, abs=0.01)
        assert result.mix_duration == pytest.approx(1.0, abs=0.01)

    def test_byte_determinism_across_two_runs(self, tmp_path):
        dataset_root = tmp_path / "root"
        _make_source(dataset_root)
        record = _make_record()

        r1 = cut_window_wavs(record, dataset_root, tmp_path / "out1")
        r2 = cut_window_wavs(record, dataset_root, tmp_path / "out2")

        for p1, p2 in zip(r1.channel_paths, r2.channel_paths):
            assert p1.read_bytes() == p2.read_bytes()
        assert r1.mix_path.read_bytes() == r2.mix_path.read_bytes()

    def test_mix_peak_stays_within_pcm16_range(self, tmp_path):
        dataset_root = tmp_path / "root"
        # Correlated tones on both channels: sum can approach clipping.
        _make_source(dataset_root, freqs=(220.0, 220.0))
        out_dir = tmp_path / "out"
        record = _make_record()

        result = cut_window_wavs(record, dataset_root, out_dir)

        mix_data, _ = sf.read(str(result.mix_path), dtype="float32")
        assert np.max(np.abs(mix_data)) <= 1.0 + 1e-3

    def test_mix_equals_sum_of_exact_written_channel_arrays(self, tmp_path):
        """DESIGN-CRITICAL (decision 12): the mix must be built from the same
        float32 arrays that get quantized into the channel wavs, not from a
        second independent resample. We compare against load_window_channel
        called directly (the same function cut_window_wavs uses internally)
        rather than re-reading the quantized channel wavs, since PCM16
        round-tripping two channels separately and then summing would not
        equal quantizing the already-summed mix. The mix wav is itself a
        PCM16 quantization of that exact sum, so we allow a 1-LSB tolerance
        (2/32768) when comparing against the file read back from disk.
        """
        dataset_root = tmp_path / "root"
        audio_path = _make_source(dataset_root)
        record = _make_record()

        ch0 = load_window_channel(audio_path, record.t0, record.t1, 0, TARGET_SR)
        ch1 = load_window_channel(audio_path, record.t0, record.t1, 1, TARGET_SR)
        expected_mix = mix_mono([ch0, ch1])

        out_dir = tmp_path / "out"
        result = cut_window_wavs(record, dataset_root, out_dir)
        mix_data, _ = sf.read(str(result.mix_path), dtype="float32")

        np.testing.assert_allclose(mix_data, expected_mix, atol=2 / 32768)

    def test_mix_is_not_a_separate_resample_call(self, tmp_path):
        """Exactly one soxr.resample call per channel, none extra for the
        mix: mix_mono only sums arrays already produced by
        load_window_channel."""
        dataset_root = tmp_path / "root"
        _make_source(dataset_root)
        out_dir = tmp_path / "out"
        record = _make_record()

        with patch(
            "dataset.preprocessing.audio.soxr.resample", wraps=soxr.resample
        ) as spy:
            cut_window_wavs(record, dataset_root, out_dir)

        assert spy.call_count == record.num_channels
