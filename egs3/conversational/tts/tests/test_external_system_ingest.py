"""Tests for ``src/external_system_ingest.py``: ingesting ANOTHER system's
audio into our output contract so the ordinary measure stage scores it.

Fixture-based and CPU-only; reuses the external-manifest fixtures, since a
baseline is measured against exactly the manifest our own arms read.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from egs3.conversational.tts.src.external_system_ingest import MODE as INGEST_MODE
from egs3.conversational.tts.src.external_system_ingest import (
    run_external_system_ingest,
)
from egs3.conversational.tts.src.system import INGEST_MODE as SYSTEM_INGEST_MODE

from .test_external_manifest import (
    FS,
    ONE_SPK,
    TWO_SPK,
    _meta,
    _read_wav,
    write_manifest,
)

SYSTEM = {"name": "fake-baseline", "repo": "https://example.invalid/fake"}


def _tracks(seconds: float, onsets, sr: int = FS, freq: float = 400.0) -> np.ndarray:
    """``(T, C)`` of tone tracks, track ``c`` silent until ``onsets[c]`` sec."""
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    out = np.zeros((n, len(onsets)), dtype=np.float32)
    for ch, onset in enumerate(onsets):
        start = int(round(onset * sr))
        tone = 0.2 * np.sin(2 * np.pi * freq * (ch + 1) * t[start:])
        out[start:, ch] = tone.astype(np.float32)
    return out


def _write_system_wav(wav_dir, wid, seconds, onsets, sr=FS):
    wav_dir.mkdir(parents=True, exist_ok=True)
    data = _tracks(seconds, onsets, sr=sr)
    sf.write(str(wav_dir / f"{wid}.wav"), data, sr, subtype="PCM_16")
    return data


def _config(fx, inference_dir, wav_dir, **ingest_overrides):
    ingest = {
        "wav_dir": str(wav_dir),
        "suffix": ".wav",
        "mono_extra_track": "require_silent",
        "verify_channel_map": True,
        "channel_order": None,
    }
    ingest.update(ingest_overrides)
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "test_name": "valid",
            "mode": INGEST_MODE,
            "device": "cpu",
            "testset": {
                "manifest": str(fx["manifest"]),
                "name": "zipvoice-dialog-test-en-v2",
            },
            "system": SYSTEM,
            "ingest": ingest,
            "selection": {"dialogue_ids": None},
        }
    )


def _two_spk_case(tmp_path, onsets=(0.0, 1.0), seconds=3.0, sr=FS):
    fx = write_manifest(tmp_path, [TWO_SPK])
    wav_dir = tmp_path / "system"
    data = _write_system_wav(wav_dir, "d2", seconds, onsets, sr=sr)
    return fx, wav_dir, data


class TestIngest:
    def test_dispatch_literal_matches_mode(self):
        assert SYSTEM_INGEST_MODE == INGEST_MODE == "ingest_external_system"

    def test_writes_the_measure_contract(self, tmp_path):
        fx, wav_dir, data = _two_spk_case(tmp_path)
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        assert stats["n_selected"] == 1
        assert stats["n_skipped"] == 0
        test_dir = tmp_path / "out" / "valid"
        meta = _meta(test_dir, "d2")
        assert meta["mode"] == INGEST_MODE
        assert meta["system"] == SYSTEM
        assert meta["num_channels"] == 2
        assert meta["has_reference_audio"] is True
        # The baseline's own length, recorded as such: no rule, no oracle.
        assert meta["duration"]["source"] == "system"
        assert meta["duration"]["predicted_sec"] == pytest.approx(3.0, abs=1 / FS)
        assert meta["duration"]["gt_sec"] == pytest.approx(4.0, abs=1e-6)
        assert meta["window_duration_sec"] == pytest.approx(3.0, abs=1 / FS)
        for ch in range(2):
            gen, sr = _read_wav(test_dir / meta["channels"][ch]["gen_wav"])
            assert sr == FS
            # PCM_16 round-trip, so compare at 16-bit resolution.
            assert np.allclose(gen, data[:, ch], atol=2e-4)
            assert (test_dir / meta["channels"][ch]["gt_wav"]).is_file()
            assert (test_dir / meta["channels"][ch]["prompt_wav"]).is_file()
            assert meta["channels"][ch]["ref_text"]
        mix, _ = _read_wav(test_dir / meta["mix_wav"])
        assert np.allclose(mix, data.mean(axis=1), atol=2e-4)
        for name in (
            "meta.scp",
            "wav.scp",
            "prompt.scp",
            "text.scp",
            "mix.scp",
            "gt.scp",
        ):
            assert (test_dir / name).is_file()

    def test_resamples_to_the_training_rate(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path, seconds=3.0, sr=16000)
        run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        test_dir = tmp_path / "out" / "valid"
        meta = _meta(test_dir, "d2")
        gen, sr = _read_wav(test_dir / meta["channels"][0]["gen_wav"])
        assert sr == FS
        assert gen.shape[0] == pytest.approx(3.0 * FS, rel=1e-3)

    def test_missing_output_is_reported_with_the_ids(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        wav_dir = tmp_path / "system"
        wav_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="d2"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", wav_dir),
                training_config=fx["training_config"],
            )

    def test_too_few_tracks_is_an_error(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        wav_dir = tmp_path / "system"
        _write_system_wav(wav_dir, "d2", 3.0, (0.0,))
        with pytest.raises(ValueError, match="1 track"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", wav_dir),
                training_config=fx["training_config"],
            )

    def test_system_name_is_required(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path)
        cfg = _config(fx, tmp_path / "out", wav_dir)
        cfg.system = {}
        with pytest.raises(ValueError, match="system.name"):
            run_external_system_ingest(cfg, training_config=fx["training_config"])

    def test_unknown_wav_dir_is_an_error(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        with pytest.raises(FileNotFoundError, match="wav_dir"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", tmp_path / "nope"),
                training_config=fx["training_config"],
            )


class TestMonoRecords:
    """A 1-channel record against a stereo-only baseline: the extra track is
    the model's answer to a speaker the transcript never mentions."""

    def test_silent_extra_track_is_accepted_and_dropped(self, tmp_path):
        fx = write_manifest(tmp_path, [ONE_SPK])
        wav_dir = tmp_path / "system"
        data = _tracks(2.0, (0.0, 0.0))
        data[:, 1] = 0.0
        wav_dir.mkdir()
        sf.write(str(wav_dir / "d1.wav"), data, FS, subtype="PCM_16")
        run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        meta = _meta(tmp_path / "out" / "valid", "d1")
        assert meta["num_channels"] == 1
        assert len(meta["channels"]) == 1
        # Never verifiable with one channel: there is no ordering to compare.
        assert meta["channel_map"] == "unverifiable"

    def test_speech_on_the_unused_track_is_a_finding(self, tmp_path):
        fx = write_manifest(tmp_path, [ONE_SPK])
        wav_dir = tmp_path / "system"
        _write_system_wav(wav_dir, "d1", 2.0, (0.0, 0.0))
        with pytest.raises(ValueError, match="not silent"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", wav_dir),
                training_config=fx["training_config"],
            )

    def test_ignore_policy_accepts_it(self, tmp_path):
        fx = write_manifest(tmp_path, [ONE_SPK])
        wav_dir = tmp_path / "system"
        _write_system_wav(wav_dir, "d1", 2.0, (0.0, 0.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir, mono_extra_track="ignore"),
            training_config=fx["training_config"],
        )
        assert stats["n_selected"] == 1

    def test_unknown_policy_is_rejected(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path)
        with pytest.raises(ValueError, match="mono_extra_track"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", wav_dir, mono_extra_track="maybe"),
                training_config=fx["training_config"],
            )


class TestChannelMap:
    """Which track carries which speaker is verified, never assumed: a silent
    swap would corrupt every per-channel row."""

    def test_expected_order_reads_as_is(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path, onsets=(0.0, 1.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        assert stats["channel_map"] == {
            "as_is": 1,
            "swapped": 0,
            "unverifiable": 0,
        }
        assert _meta(tmp_path / "out" / "valid", "d2")["channel_map"] == "as_is"

    def test_reversed_onsets_read_as_swapped_without_reordering(self, tmp_path):
        fx, wav_dir, data = _two_spk_case(tmp_path, onsets=(1.0, 0.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        assert stats["channel_map"]["swapped"] == 1
        test_dir = tmp_path / "out" / "valid"
        meta = _meta(test_dir, "d2")
        assert meta["channel_map"] == "swapped"
        # Reported, NOT silently fixed: channel 0 is still track 0.
        gen, _ = _read_wav(test_dir / meta["channels"][0]["gen_wav"])
        assert np.allclose(gen, data[:, 0], atol=2e-4)

    def test_channel_order_reorders_the_tracks(self, tmp_path):
        fx, wav_dir, data = _two_spk_case(tmp_path, onsets=(1.0, 0.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir, channel_order=[1, 0]),
            training_config=fx["training_config"],
        )
        assert stats["channel_map"]["as_is"] == 1
        test_dir = tmp_path / "out" / "valid"
        meta = _meta(test_dir, "d2")
        gen, _ = _read_wav(test_dir / meta["channels"][0]["gen_wav"])
        assert np.allclose(gen, data[:, 1], atol=2e-4)

    def test_bad_channel_order_is_rejected(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path)
        with pytest.raises(ValueError, match="permutation"):
            run_external_system_ingest(
                _config(fx, tmp_path / "out", wav_dir, channel_order=[0, 0]),
                training_config=fx["training_config"],
            )

    def test_simultaneous_onsets_are_unverifiable(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path, onsets=(0.0, 0.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir),
            training_config=fx["training_config"],
        )
        assert stats["channel_map"]["unverifiable"] == 1

    def test_verification_can_be_turned_off(self, tmp_path):
        fx, wav_dir, _ = _two_spk_case(tmp_path, onsets=(1.0, 0.0))
        stats = run_external_system_ingest(
            _config(fx, tmp_path / "out", wav_dir, verify_channel_map=False),
            training_config=fx["training_config"],
        )
        assert stats["channel_map"]["swapped"] == 0
        assert stats["channel_map"]["unverifiable"] == 1


def test_meta_is_json_and_carries_provenance(tmp_path):
    fx, wav_dir, _ = _two_spk_case(tmp_path)
    run_external_system_ingest(
        _config(fx, tmp_path / "out", wav_dir),
        training_config=fx["training_config"],
    )
    raw = (tmp_path / "out" / "valid" / "meta" / "d2.json").read_text("utf-8")
    meta = json.loads(raw)
    assert meta["system"]["name"] == "fake-baseline"
    assert meta["testset"] == "zipvoice-dialog-test-en-v2"
