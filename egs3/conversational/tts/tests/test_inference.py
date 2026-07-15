"""Infer-stage tests: prompt-boundary snapping, the meta/SCP output contract,
and generate/gt/resynth layout parity.

Fixture-based and CPU-only: a fabricated two-channel FLAC + a hand-built window
manifest, the tiny random-init DiT from the trainer suite, and a fake Vocos
whose ``decode`` maps a mel ``(N, n_mel, T)`` to a wave ``(N, T*hop)``.  ``gt``
mode needs neither model nor vocoder (pure audio slicing), so its meta JSON is
compared byte-for-byte against a golden dict.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from .conftest import EXT_TOKENS
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    to_json,
)
from egs3.conversational.tts.src.inference import (
    run_inference,
    snap_prompt_boundary,
)

FS = 24000
SRC_SR = 48000
HOP = 256

FakeTurn = namedtuple("FakeTurn", ["channel", "speaker", "text", "start", "end"])


# --------------------------------------------------------------------------- #
# Prompt-boundary snapping (pure function)
# --------------------------------------------------------------------------- #
class TestSnapping:
    def _turns(self):
        # window-relative endpoints (t0 subtracted) are stated in comments.
        return [
            FakeTurn(0, "a", "abc def", 5.5, 8.0),  # rel 0.5 - 3.0
            FakeTurn(1, "b", "bead cab", 8.4, 10.5),  # rel 3.4 - 5.5
            FakeTurn(0, "a", "fade dad", 11.0, 13.5),  # rel 6.0 - 8.5
            FakeTurn(1, "b", "bag", 14.0, 15.0),  # rel 9.0 - 10.0
        ]

    def test_closest_eligible_boundary_in_band(self):
        b = snap_prompt_boundary(
            self._turns(), t0=5.0, target_sec=3.0, prompt_min=2.0, prompt_max=10.0
        )
        assert b == pytest.approx(3.0)  # ch0 turn-1 end, exactly on target

    def test_no_turn_strictly_contains_the_cut(self):
        turns = self._turns()
        b = snap_prompt_boundary(
            turns, t0=5.0, target_sec=4.5, prompt_min=2.0, prompt_max=10.0
        )
        cut_abs = 5.0 + b
        for t in turns:
            assert not (t.start < cut_abs < t.end)

    def test_target_in_silence_snaps_to_a_turn_endpoint(self):
        # target 4.2 (abs 9.2) sits in the silence between ch1-turn1 (ends 5.5)
        # and ch0-turn2 (starts 6.0); nearest endpoints are 3.4 and 5.5.
        turns = self._turns()
        b = snap_prompt_boundary(
            turns, t0=5.0, target_sec=4.2, prompt_min=2.0, prompt_max=10.0
        )
        endpoints = {round(e - 5.0, 6) for t in turns for e in (t.start, t.end)}
        assert b in endpoints

    def test_cross_channel_overlap_makes_endpoint_ineligible(self):
        # ch0 turn ends at 3.0; a ch1 turn spans (2.5, 4.0) so the instant 3.0
        # is strictly inside it -> ineligible.  Only 4.0 (abs) survives.
        turns = [
            FakeTurn(0, "a", "abc", 0.5, 3.0),  # rel 3.0 endpoint, blocked
            FakeTurn(1, "b", "bead", 2.5, 4.0),  # rel 4.0 endpoint, eligible
        ]
        b = snap_prompt_boundary(
            turns, t0=0.0, target_sec=3.0, prompt_min=2.0, prompt_max=6.0
        )
        assert b == pytest.approx(4.0)

    def test_no_eligible_boundary_in_band_returns_none(self):
        turns = self._turns()
        # band above every endpoint -> nothing eligible in [10.5, 12.0]
        b = snap_prompt_boundary(
            turns, t0=5.0, target_sec=11.0, prompt_min=10.5, prompt_max=12.0
        )
        assert b is None

    def test_tie_breaks_toward_earlier(self):
        turns = [
            FakeTurn(0, "a", "abc", 0.0, 2.0),  # rel 2.0
            FakeTurn(1, "b", "bea", 4.0, 6.0),  # rel 4.0
        ]
        # target 3.0 is equidistant from 2.0 and 4.0 -> earlier (2.0) wins.
        b = snap_prompt_boundary(
            turns, t0=0.0, target_sec=3.0, prompt_min=1.0, prompt_max=7.0
        )
        assert b == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Fixtures for full-stage runs
# --------------------------------------------------------------------------- #
def _write_flac(path: Path, num_channels: int, duration_s: float, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    data = np.stack(
        [0.2 * np.sin(2 * 3.14159 * 400 * (c + 1) * t) for c in range(num_channels)],
        axis=1,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="FLAC")


def _window():
    return WindowRecord(
        window_id="sess_w00000",
        session_id="sess",
        audio_relpath="original/sess_mixed.flac",
        num_channels=2,
        sample_rate=SRC_SR,
        t0=5.0,
        t1=17.0,
        turns=(
            Turn(0, "spk_a", "abc def", 5.5, 8.0),
            Turn(1, "spk_b", "bead cab", 8.4, 10.5),
            Turn(0, "spk_a", "fade dad", 11.0, 13.5),
            Turn(1, "spk_b", "bag", 14.0, 15.0),
        ),
    )


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "data"
    _write_flac(root / "original" / "sess_mixed.flac", 2, 20.0, SRC_SR)
    manifest = root / "valid.jsonl"
    manifest.write_text(json.dumps(to_json(_window())) + "\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    training_config = OmegaConf.create(
        {
            "recipe_dir": str(tmp_path),
            "sample_rate": FS,
            "hop_length": HOP,
            "dataset": {"preprocessor": {"token_list": str(vocab)}},
        }
    )
    return {
        "tmp_path": tmp_path,
        "manifest": manifest,
        "dataset_root": root,
        "vocab": vocab,
        "training_config": training_config,
    }


def _infer_config(fixture, mode, inference_dir):
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "test_name": "valid",
            "mode": mode,
            "device": "cpu",
            "ckpt": None,
            "use_ema": True,
            "dataset": {
                "split": "valid",
                "manifest_path": str(fixture["manifest"]),
                "dataset_root": str(fixture["dataset_root"]),
            },
            "selection": {
                "num_active_speakers": 2,
                "min_duration": None,
                "max_duration": None,
                "num_windows": 10,
                "seed": 0,
            },
            "prompt": {
                "target_sec": 3.0,
                "min_sec": 2.0,
                "max_sec": 10.0,
                "boundary_guard": 0.0,
            },
            "sampling": {
                "steps": 2,
                "cfg_strength": 2.0,
                "sway_sampling_coef": -1.0,
                "seed": 0,
            },
        }
    )


class FakeVocoder:
    """Deterministic stand-in for Vocos: mel ``(N, n_mel, T)`` -> ``(N, T*hop)``."""

    def __init__(self, hop: int = HOP):
        self.hop = hop

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        n, _, t = mel.shape
        # Mean over mel bins, upsampled by hop, tanh-bounded: finite audio that
        # varies with the input so resynth != silence.
        frame = torch.tanh(mel.mean(dim=1))  # (N, T)
        return frame.repeat_interleave(self.hop, dim=1)  # (N, T*hop)


def _read_wav(path: Path):
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data, sr


# --------------------------------------------------------------------------- #
# gt-mode golden contract
# --------------------------------------------------------------------------- #
class TestGtContract:
    def _run(self, fixture):
        inf_dir = fixture["tmp_path"] / "infer"
        cfg = _infer_config(fixture, "gt", inf_dir)
        stats = run_inference(cfg, training_config=fixture["training_config"])
        return inf_dir / "valid", stats

    def test_meta_scp_and_golden_json(self, fixture):
        test_dir, stats = self._run(fixture)
        assert stats["n_selected"] == 1
        assert stats["n_skipped"] == 0

        scp = (test_dir / "meta.scp").read_text(encoding="utf-8").splitlines()
        assert scp == ["sess_w00000 meta/sess_w00000.json"]

        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        # Hardcoded, not `round(3.0 * FS) // HOP`: the golden value must not
        # be derived with the same arithmetic the code under test uses, or a
        # shared bug in that formula would pass both sides silently.
        # FS=24000, HOP=256 -> round(3.0 * 24000) // 256 == 72000 // 256 == 281.
        prompt_frames = 281
        expected = {
            "window_id": "sess_w00000",
            "session_id": "sess",
            "mode": "gt",
            "sample_rate": FS,
            "num_channels": 2,
            "prompt_boundary_sec": 3.0,
            "prompt_boundary_frames": prompt_frames,
            "window_duration_sec": 12.0,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": "wav/sess_w00000_ch0.wav",
                    "prompt_wav": "prompt/sess_w00000_ch0.wav",
                    "gt_wav": "gt/sess_w00000_ch0.wav",
                    "ref_text": "fade dad",
                },
                {
                    "gen_wav": "wav/sess_w00000_ch1.wav",
                    "prompt_wav": "prompt/sess_w00000_ch1.wav",
                    "gt_wav": "gt/sess_w00000_ch1.wav",
                    "ref_text": "bead cab bag",
                },
            ],
            "turns": [
                {"channel": 0, "text": "abc def", "start": 0.5, "end": 3.0},
                {"channel": 1, "text": "bead cab", "start": 3.4, "end": 5.5},
                {"channel": 0, "text": "fade dad", "start": 6.0, "end": 8.5},
                {"channel": 1, "text": "bag", "start": 9.0, "end": 10.0},
            ],
        }
        assert meta == expected

    def test_relative_paths_resolve_and_open(self, fixture):
        test_dir, _ = self._run(fixture)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        for ch in meta["channels"]:
            for key in ("gen_wav", "prompt_wav", "gt_wav"):
                data, sr = _read_wav(test_dir / ch[key])
                assert sr == FS
                assert data.size > 0

    def test_gt_generated_equals_gt_reference(self, fixture):
        # In gt mode the "generated" wav IS the ground-truth region.
        test_dir, _ = self._run(fixture)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        ch = meta["channels"][0]
        gen, _ = _read_wav(test_dir / ch["gen_wav"])
        gt, _ = _read_wav(test_dir / ch["gt_wav"])
        assert (gen == gt).all()

    def test_prompt_and_generated_lengths(self, fixture):
        test_dir, _ = self._run(fixture)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        prompt_samples = meta["prompt_boundary_frames"] * HOP
        ch = meta["channels"][0]
        prompt, _ = _read_wav(test_dir / ch["prompt_wav"])
        assert prompt.shape[0] == prompt_samples

    def test_convenience_scps(self, fixture):
        test_dir, _ = self._run(fixture)
        wav = (test_dir / "wav.scp").read_text("utf-8").splitlines()
        prompt = (test_dir / "prompt.scp").read_text("utf-8").splitlines()
        text = (test_dir / "text.scp").read_text("utf-8").splitlines()
        mix = (test_dir / "mix.scp").read_text("utf-8").splitlines()
        assert [ln.split(maxsplit=1)[0] for ln in wav] == [
            "sess_w00000_ch0",
            "sess_w00000_ch1",
        ]
        assert [ln.split(maxsplit=1)[0] for ln in prompt] == [
            "sess_w00000_ch0",
            "sess_w00000_ch1",
        ]
        assert text[0] == "sess_w00000_ch0 fade dad"
        assert text[1] == "sess_w00000_ch1 bead cab bag"
        assert mix == ["sess_w00000 mix/sess_w00000.wav"]


# --------------------------------------------------------------------------- #
# generate / gt / resynth layout parity
# --------------------------------------------------------------------------- #
class TestModeParity:
    def _run_mode(self, fixture, mode, ext_vocab_file):
        inf_dir = fixture["tmp_path"] / f"infer_{mode}"
        cfg = _infer_config(fixture, mode, inf_dir)
        model = None
        vocoder = None
        if mode in ("generate", "resynth"):
            model = build_tiny(ext_vocab_file).eval()
            vocoder = FakeVocoder()
        run_inference(
            cfg,
            training_config=fixture["training_config"],
            model=model,
            vocoder=vocoder,
        )
        return inf_dir / "valid"

    def _layout(self, test_dir: Path):
        files = sorted(
            str(p.relative_to(test_dir))
            for p in test_dir.rglob("*")
            if p.is_file()
        )
        return files

    def test_layout_identical_across_modes(self, fixture, ext_vocab_file):
        gt_dir = self._run_mode(fixture, "gt", ext_vocab_file)
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        res_dir = self._run_mode(fixture, "resynth", ext_vocab_file)
        assert self._layout(gt_dir) == self._layout(gen_dir) == self._layout(res_dir)

    def test_meta_keys_identical_across_modes(self, fixture, ext_vocab_file):
        gt_dir = self._run_mode(fixture, "gt", ext_vocab_file)
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        gt_meta = json.loads((gt_dir / "meta/sess_w00000.json").read_text("utf-8"))
        gen_meta = json.loads((gen_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert set(gt_meta) == set(gen_meta)
        assert [set(c) for c in gt_meta["channels"]] == [
            set(c) for c in gen_meta["channels"]
        ]
        # Boundary/text/turns are mode-invariant; only rtf and audio differ.
        for key in ("prompt_boundary_sec", "prompt_boundary_frames", "turns"):
            assert gt_meta[key] == gen_meta[key]

    def test_generate_reports_positive_rtf(self, fixture, ext_vocab_file):
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        meta = json.loads((gen_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert isinstance(meta["rtf"], float)
        assert meta["rtf"] > 0.0

    def test_resynth_rtf_is_null(self, fixture, ext_vocab_file):
        res_dir = self._run_mode(fixture, "resynth", ext_vocab_file)
        meta = json.loads((res_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert meta["rtf"] is None

    def test_generated_audio_is_finite(self, fixture, ext_vocab_file):
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        meta = json.loads((gen_dir / "meta/sess_w00000.json").read_text("utf-8"))
        for ch in meta["channels"]:
            data, _ = _read_wav(gen_dir / ch["gen_wav"])
            assert data.size > 0
            assert bool((abs(data) < 1e9).all())


# --------------------------------------------------------------------------- #
# selection + skip accounting
# --------------------------------------------------------------------------- #
class TestSystemDispatch:
    """The production path: ConversationalTTSSystem.infer() loading the training
    config from disk (training_config=None), as `python run.py --stages infer`."""

    def test_system_infer_loads_training_config_from_disk(self, fixture):
        from egs3.conversational.tts.src.system import ConversationalTTSSystem

        train_yaml = fixture["tmp_path"] / "train.yaml"
        OmegaConf.save(fixture["training_config"], train_yaml)

        inf_dir = fixture["tmp_path"] / "infer_dispatch"
        cfg = _infer_config(fixture, "gt", inf_dir)
        cfg.training_config = str(train_yaml)  # absolute -> loaded as-is

        system = ConversationalTTSSystem(inference_config=cfg)
        stats = system.infer()
        assert stats == {"n_selected": 1, "n_skipped": 0}

        test_dir = inf_dir / "valid"
        scp = (test_dir / "meta.scp").read_text("utf-8").splitlines()
        assert scp == ["sess_w00000 meta/sess_w00000.json"]
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert meta["mode"] == "gt"
        assert meta["prompt_boundary_sec"] == 3.0
        assert meta["channels"][1]["ref_text"] == "bead cab bag"


class TestSelection:
    def test_window_with_no_eligible_boundary_is_skipped(self, fixture):
        inf_dir = fixture["tmp_path"] / "infer_skip"
        cfg = _infer_config(fixture, "gt", inf_dir)
        # Band above every turn endpoint -> nothing eligible -> skip + count.
        cfg.prompt.min_sec = 11.0
        cfg.prompt.max_sec = 12.0
        cfg.prompt.target_sec = 11.5
        stats = run_inference(cfg, training_config=fixture["training_config"])
        assert stats["n_selected"] == 0
        assert stats["n_skipped"] == 1
        assert not (inf_dir / "valid" / "meta.scp").exists() or (
            inf_dir / "valid" / "meta.scp"
        ).read_text("utf-8").strip() == ""
