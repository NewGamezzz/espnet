"""Infer-stage tests: turn-pool construction, the prompt relaxation ladder,
the meta/SCP output contract, and generate/gt/resynth layout parity.

Fixture-based and CPU-only: a fabricated two-channel FLAC + a hand-built
TWO-WINDOW manifest on one session, the tiny random-init DiT from the trainer
suite, and a fake Vocos whose ``decode`` maps a mel ``(N, n_mel, T)`` to a
wave ``(N, T*hop)``.  Two windows per session are load-bearing: the new
scheme forbids drawing a channel's prompt turn from inside the evaluated
window (leakage), so a session with only one window has an empty candidate
pool for every channel and every window is skipped - the happy-path fixture
below gives each window's channels exactly one non-window candidate (the
other window's turns), so picks are forced and deterministic without
depending on ``random.Random`` internals.  ``gt`` mode needs neither model
nor vocoder (pure audio slicing + concatenation), so its meta JSON is
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
from egs3.conversational.tts.dataset.preprocessing.text import (
    TURN_FILL_TOKEN,
    build_branch_texts,
    encode_tokens,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    to_json,
)
from egs3.conversational.tts.src import inference as inference_mod
from egs3.conversational.tts.src.inference import (
    _build_turn_pools,
    _select_prompt_turn,
    run_inference,
)

FS = 24000
SRC_SR = 48000
HOP = 256

# Mode T emits <turn_fill>, the fifth (timestamp-era) new token; the shared
# `ext_vocab_file` / EXT_TOKENS fixtures stop at the four special-token-era
# ones, so the Mode T tests pair a 5-token vocab with a model sized for it.
EXT5_TOKENS = EXT_TOKENS + [TURN_FILL_TOKEN]

_RecordStub = namedtuple("_RecordStub", ["session_id", "turns"])
_TextTurn = namedtuple("_TextTurn", ["channel", "text"])


# --------------------------------------------------------------------------- #
# Turn-pool construction (pure function)
# --------------------------------------------------------------------------- #
class TestTurnPools:
    def test_union_across_records_deduped(self):
        shared = Turn(0, "a", "abc", 1.0, 2.0)
        shared_dup = Turn(0, "a", "abc", 1.0, 2.0)  # same fields -> deduped
        other_ch1 = Turn(1, "b", "bead", 3.0, 4.0)
        unique = Turn(0, "a", "cab", 5.0, 6.0)
        records = [
            _RecordStub("sess", (shared, other_ch1)),
            _RecordStub("sess", (shared_dup, unique)),
        ]
        pools = _build_turn_pools(records)
        assert pools["sess"] == [shared, other_ch1, unique]

    def test_sessions_kept_separate(self):
        ta = Turn(0, "a", "abc", 1.0, 2.0)
        tb = Turn(0, "a", "abc", 1.0, 2.0)  # same fields, different session
        records = [_RecordStub("sess1", (ta,)), _RecordStub("sess2", (tb,))]
        pools = _build_turn_pools(records)
        assert pools["sess1"] == [ta]
        assert pools["sess2"] == [tb]

    def test_empty_records_give_empty_pools_dict(self):
        assert _build_turn_pools([]) == {}


# --------------------------------------------------------------------------- #
# Relaxation ladder (pure function)
# --------------------------------------------------------------------------- #
class TestPromptTurnLadder:
    def test_leakage_exclusion_is_absolute(self):
        # The channel's only turn lies inside the evaluated window -> None
        # regardless of every relaxation tier.
        turns = [Turn(0, "a", "x", 1.0, 3.0)]
        result = _select_prompt_turn(
            turns,
            0,
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        assert result is None

    def test_solo_preferred_over_overlapped(self):
        overlapped = Turn(0, "a", "x", 10.0, 13.0)
        overlap_partner = Turn(1, "b", "y", 11.0, 14.0)  # collides w/ overlapped
        solo = Turn(0, "a", "z", 20.0, 23.0)
        pool = [overlapped, overlap_partner, solo]
        result = _select_prompt_turn(
            pool,
            0,
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        assert result is solo

    def test_band_preferred_over_out_of_band(self):
        too_short = Turn(0, "a", "x", 10.0, 11.0)  # 1.0s, below turn_min
        in_band = Turn(0, "a", "y", 20.0, 23.0)  # 3.0s
        pool = [too_short, in_band]
        result = _select_prompt_turn(
            pool,
            0,
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        assert result is in_band

    def test_relaxes_to_solo_when_band_is_empty(self):
        # Only candidate is solo but out of band -> tier 2 still returns it.
        too_short = Turn(0, "a", "x", 10.0, 11.0)
        result = _select_prompt_turn(
            [too_short],
            0,
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        assert result is too_short

    def test_relaxes_to_non_window_when_no_solo_candidate_exists(self):
        overlapped = Turn(0, "a", "x", 10.0, 13.0)
        overlap_partner = Turn(1, "b", "y", 11.0, 14.0)
        result = _select_prompt_turn(
            [overlapped, overlap_partner],
            0,
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        assert result is overlapped

    def test_deterministic_pick_given_seed(self):
        a = Turn(0, "a", "x", 10.0, 13.0)
        b = Turn(0, "a", "y", 20.0, 23.0)
        pool = [a, b]  # both solo, both in band -> tier 1 has 2 candidates
        kwargs = dict(
            t0=0.0,
            t1=5.0,
            turn_min=2.0,
            turn_max=10.0,
            seed=0,
            window_id="w",
        )
        first = _select_prompt_turn(pool, 0, **kwargs)
        second = _select_prompt_turn(pool, 0, **kwargs)
        assert first is second


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


def _window_a() -> WindowRecord:
    return WindowRecord(
        window_id="sess_w00000",
        session_id="sess",
        audio_relpath="original/sess_mixed.flac",
        num_channels=2,
        sample_rate=SRC_SR,
        t0=5.0,
        t1=13.0,
        turns=(
            Turn(0, "spk_a", "abc def", 5.5, 8.0),  # 2.5s, rel 0.5-3.0
            Turn(1, "spk_b", "bead cab", 8.5, 11.0),  # 2.5s, rel 3.5-6.0
        ),
    )


def _window_b() -> WindowRecord:
    return WindowRecord(
        window_id="sess_w00001",
        session_id="sess",
        audio_relpath="original/sess_mixed.flac",
        num_channels=2,
        sample_rate=SRC_SR,
        t0=25.0,
        t1=33.0,
        turns=(
            Turn(0, "spk_a", "cage jade", 25.5, 28.0),  # 2.5s
            Turn(1, "spk_b", "badge fig", 28.5, 31.0),  # 2.5s
        ),
    )


def _write_fixture_files(tmp_path, windows, flac_duration_s: float) -> dict:
    root = tmp_path / "data"
    _write_flac(root / "original" / "sess_mixed.flac", 2, flac_duration_s, SRC_SR)
    manifest = root / "valid.jsonl"
    manifest.write_text(
        "".join(json.dumps(to_json(w)) + "\n" for w in windows), encoding="utf-8"
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    vocab5 = tmp_path / "vocab5.txt"
    vocab5.write_text("\n".join(EXT5_TOKENS) + "\n", encoding="utf-8")

    def _training_config(token_list):
        return OmegaConf.create(
            {
                "recipe_dir": str(tmp_path),
                "sample_rate": FS,
                "hop_length": HOP,
                "dataset": {"preprocessor": {"token_list": str(token_list)}},
            }
        )

    return {
        "tmp_path": tmp_path,
        "manifest": manifest,
        "dataset_root": root,
        "vocab": vocab,
        "vocab5": vocab5,
        "training_config": _training_config(vocab),
        # Same config against the 5-token (<turn_fill>-carrying) vocab, for
        # the Mode T runs; the 4-token one above is left untouched.
        "training_config_5": _training_config(vocab5),
    }


@pytest.fixture
def ext_vocab5_file(tmp_path) -> Path:
    """The conftest `ext_vocab_file` plus <turn_fill> (Mode T's fill token)."""
    path = tmp_path / "vocab5.txt"
    path.write_text("\n".join(EXT5_TOKENS) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fixture(tmp_path):
    """Two windows on one session: each window's channels draw their prompt
    turn from the OTHER window (the only non-window candidate), so every
    ladder pick is forced to a single element regardless of the seed."""
    return _write_fixture_files(tmp_path, [_window_a(), _window_b()], 40.0)


@pytest.fixture
def solo_window_fixture(tmp_path):
    """One window, one session: every channel's only pool turns are its own
    (in-window) turns, so the non-window tier is always empty."""
    return _write_fixture_files(tmp_path, [_window_a()], 20.0)


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
                "turn_min_sec": 2.0,
                "turn_max_sec": 10.0,
                "seed": 0,
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
        assert stats["n_selected"] == 2
        assert stats["n_skipped"] == 0

        scp = (test_dir / "meta.scp").read_text(encoding="utf-8").splitlines()
        assert scp == [
            "sess_w00000 meta/sess_w00000.json",
            "sess_w00001 meta/sess_w00001.json",
        ]

        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        # Hardcoded, not derived with the same arithmetic the code under test
        # uses, so a shared bug in that formula would not pass both sides.
        # Prompt = window B's ch0 turn (2.5s) + ch1 turn (2.5s) = 120000
        # samples at FS=24000; 120000 // HOP(256) == 468 (remainder 192
        # dropped from the end). 468 * 256 == 119808 samples == 4.992s.
        prompt_frames = 468
        prompt_sec = 4.992
        expected = {
            "window_id": "sess_w00000",
            "session_id": "sess",
            "mode": "gt",
            "sample_rate": FS,
            "num_channels": 2,
            "window_duration_sec": 8.0,
            # Always present, in every mode: `TestModeParity` requires the
            # gt/generate meta key sets to match, and only Mode T adds the
            # sibling "layout" block.
            "text_format": "order",
            "rtf": None,
            "mix_wav": "mix/sess_w00000.wav",
            "prompt": {
                "total_sec": prompt_sec,
                "total_frames": prompt_frames,
                "turns": [
                    {
                        "channel": 0,
                        "text": "cage jade",
                        "start": 25.5,
                        "end": 28.0,
                        "duration_sec": 2.5,
                    },
                    {
                        "channel": 1,
                        "text": "badge fig",
                        "start": 28.5,
                        "end": 31.0,
                        "duration_sec": 2.5,
                    },
                ],
            },
            "channels": [
                {
                    "gen_wav": "wav/sess_w00000_ch0.wav",
                    "prompt_wav": "prompt/sess_w00000_ch0.wav",
                    "gt_wav": "gt/sess_w00000_ch0.wav",
                    "ref_text": "abc def",
                },
                {
                    "gen_wav": "wav/sess_w00000_ch1.wav",
                    "prompt_wav": "prompt/sess_w00000_ch1.wav",
                    "gt_wav": "gt/sess_w00000_ch1.wav",
                    "ref_text": "bead cab",
                },
            ],
            "turns": [
                {"channel": 0, "text": "abc def", "start": 0.5, "end": 3.0},
                {"channel": 1, "text": "bead cab", "start": 3.5, "end": 6.0},
            ],
        }
        assert meta == expected

    def test_relative_paths_resolve_and_open(self, fixture):
        test_dir, _ = self._run(fixture)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        for key in ("mix_wav",):
            data, sr = _read_wav(test_dir / meta[key])
            assert sr == FS
            assert data.size > 0
        for ch in meta["channels"]:
            for key in ("gen_wav", "prompt_wav", "gt_wav"):
                data, sr = _read_wav(test_dir / ch[key])
                assert sr == FS
                assert data.size > 0

    def test_gt_generated_equals_gt_reference(self, fixture):
        # In gt mode the "generated" wav IS the whole ground-truth window.
        test_dir, _ = self._run(fixture)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        ch = meta["channels"][0]
        gen, _ = _read_wav(test_dir / ch["gen_wav"])
        gt, _ = _read_wav(test_dir / ch["gt_wav"])
        assert (gen == gt).all()
        assert gen.shape[0] == round(8.0 * FS)  # the FULL window now

    def test_convenience_scps(self, fixture):
        test_dir, _ = self._run(fixture)
        wav = (test_dir / "wav.scp").read_text("utf-8").splitlines()
        prompt = (test_dir / "prompt.scp").read_text("utf-8").splitlines()
        text = (test_dir / "text.scp").read_text("utf-8").splitlines()
        mix = (test_dir / "mix.scp").read_text("utf-8").splitlines()
        assert [ln.split(maxsplit=1)[0] for ln in wav] == [
            "sess_w00000_ch0",
            "sess_w00000_ch1",
            "sess_w00001_ch0",
            "sess_w00001_ch1",
        ]
        assert [ln.split(maxsplit=1)[0] for ln in prompt] == [
            "sess_w00000_ch0",
            "sess_w00000_ch1",
            "sess_w00001_ch0",
            "sess_w00001_ch1",
        ]
        assert text[0] == "sess_w00000_ch0 abc def"
        assert text[1] == "sess_w00000_ch1 bead cab"
        assert mix == [
            "sess_w00000 mix/sess_w00000.wav",
            "sess_w00001 mix/sess_w00001.wav",
        ]


# --------------------------------------------------------------------------- #
# Audio assembly: block layout, frame-exact trim, prompt.scp sample counts
# --------------------------------------------------------------------------- #
class TestAudioAssembly:
    def test_prompt_wav_is_the_owning_channels_solo_block(self, fixture):
        inf_dir = fixture["tmp_path"] / "infer_audio"
        cfg = _infer_config(fixture, "gt", inf_dir)
        run_inference(cfg, training_config=fixture["training_config"])
        test_dir = inf_dir / "valid"

        # Window A's prompt turns are window B's ch0/ch1 turns, both 2.5s ->
        # 2.5 * FS samples per channel's own (untrimmed) turn block.
        for ch in (0, 1):
            data, sr = _read_wav(test_dir / f"prompt/sess_w00000_ch{ch}.wav")
            assert sr == FS
            assert data.shape[0] == round(2.5 * FS)

    def test_prompt_frame_exact_trim_reconstructed_from_channel_blocks(self, fixture):
        inf_dir = fixture["tmp_path"] / "infer_trim"
        cfg = _infer_config(fixture, "gt", inf_dir)
        run_inference(cfg, training_config=fixture["training_config"])
        test_dir = inf_dir / "valid"
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))

        # The concatenated prompt itself isn't written to disk, but its
        # length is the sum of the per-channel block lengths (ch0's turn
        # block, then ch1's), and meta["prompt"] must match a frame-exact
        # floor-trim of that sum.
        total_samples = sum(
            _read_wav(test_dir / f"prompt/sess_w00000_ch{ch}.wav")[0].shape[0]
            for ch in (0, 1)
        )
        expected_frames = total_samples // HOP
        assert meta["prompt"]["total_frames"] == expected_frames
        assert meta["prompt"]["total_sec"] == round((expected_frames * HOP) / FS, 6)

    def test_generated_region_covers_the_whole_window(self, fixture, ext_vocab_file):
        inf_dir = fixture["tmp_path"] / "infer_gen_len"
        cfg = _infer_config(fixture, "generate", inf_dir)
        model = build_tiny(ext_vocab_file).eval()
        vocoder = FakeVocoder()
        run_inference(
            cfg,
            training_config=fixture["training_config"],
            model=model,
            vocoder=vocoder,
        )
        test_dir = inf_dir / "valid"
        data, _ = _read_wav(test_dir / "wav/sess_w00000_ch0.wav")
        # prompt_frames=468, total_frames=1218 (both computed from
        # hop-exact fixture durations) -> generated frames = 750 -> 750*HOP
        # samples, exactly the window's own duration (8.0s @ FS).
        assert data.shape[0] == 750 * HOP == round(8.0 * FS)


# --------------------------------------------------------------------------- #
# Text assembly: model text == build_branch_texts(prompt turns + window turns)
# --------------------------------------------------------------------------- #
class TestTextAssembly:
    def test_model_text_matches_prompt_plus_window_turns(
        self, fixture, ext_vocab_file, monkeypatch
    ):
        # Both fixture windows run through generate mode, so capture every
        # call's text and match window A (sess_w00000) up by call order:
        # `_select_indices` is sorted ascending and window A is the first
        # manifest record, so it is generated first.
        captured: list = []
        orig = inference_mod.generate_region

        def spy(model, vocoder, speech, text, prompt_frames, total_frames, **kw):
            captured.append(text.clone())
            return orig(model, vocoder, speech, text, prompt_frames, total_frames, **kw)

        monkeypatch.setattr(inference_mod, "generate_region", spy)

        inf_dir = fixture["tmp_path"] / "infer_text"
        cfg = _infer_config(fixture, "generate", inf_dir)
        model = build_tiny(ext_vocab_file).eval()
        vocoder = FakeVocoder()
        run_inference(
            cfg,
            training_config=fixture["training_config"],
            model=model,
            vocoder=vocoder,
        )

        meta = json.loads(
            (inf_dir / "valid" / "meta/sess_w00000.json").read_text("utf-8")
        )
        prompt_turns = [
            _TextTurn(p["channel"], p["text"]) for p in meta["prompt"]["turns"]
        ]
        window_turns = [_TextTurn(t["channel"], t["text"]) for t in meta["turns"]]
        expected_branches = build_branch_texts(prompt_turns + window_turns, 2)
        token2id = make_token2id(EXT_TOKENS)
        expected_ids = [encode_tokens(b, token2id) for b in expected_branches]

        assert len(captured) == 2  # both fixture windows go through generate mode
        text = captured[0]  # window A, generated first (sorted index order)
        assert text.shape[0] == 2
        for ch, expected in enumerate(expected_ids):
            row = text[ch].tolist()
            body, pad = row[: len(expected)], row[len(expected) :]
            assert body == expected
            assert all(v == -1 for v in pad)


# --------------------------------------------------------------------------- #
# text_format: timestamps (Mode T) over the whole prompt+window sequence
# --------------------------------------------------------------------------- #
class TestTimestampGenerate:
    def _cfg(self, fixture, inf_dir):
        cfg = _infer_config(fixture, "generate", inf_dir)
        cfg.text_format = "timestamps"
        return cfg

    def test_text_is_frame_aligned_over_prompt_and_window(
        self, fixture, ext_vocab5_file, monkeypatch
    ):
        captured = []
        orig = inference_mod.generate_region

        def spy(model, vocoder, speech, text, prompt_frames, total_frames, **kw):
            captured.append((text.clone(), prompt_frames, total_frames))
            return orig(model, vocoder, speech, text, prompt_frames, total_frames, **kw)

        monkeypatch.setattr(inference_mod, "generate_region", spy)
        inf_dir = fixture["tmp_path"] / "infer_t"
        stats = run_inference(
            self._cfg(fixture, inf_dir),
            training_config=fixture["training_config_5"],
            model=build_tiny(ext_vocab5_file).eval(),
            vocoder=FakeVocoder(),
        )
        assert stats["n_timestamp_degraded"] == 0
        text, prompt_frames, total_frames = captured[0]  # window A
        token2id = make_token2id(read_vocab(ext_vocab5_file))
        tn, tf = token2id["<turn>"], token2id["<turn_fill>"]
        assert text.shape == (2, total_frames)
        assert prompt_frames == 468 and total_frames == 1218
        # prompt blocks: ch0 [0, 234), ch1 [234, 469); window turns shifted by 5.0 s
        assert text[0, 0] == tn and text[1, 234] == tn
        assert text[0, round(5.5 * 93.75)] == tn  # "abc def" rel 0.5 -> 516
        assert text[1, round(8.5 * 93.75)] == tn  # "bead cab" rel 3.5 -> 797
        assert (text[0] == tf).sum() > 0 and (text[1] == tf).sum() > 0
        meta = json.loads((inf_dir / "valid/meta/sess_w00000.json").read_text("utf-8"))
        assert meta["text_format"] == "timestamps"
        assert meta["layout"]["turns"][2] == {"channel": 0, "start": 5.5, "end": 8.0}

    def test_order_mode_meta_and_parity(self, fixture, ext_vocab_file):
        inf_a = fixture["tmp_path"] / "a"
        inf_b = fixture["tmp_path"] / "b"
        cfg_a = _infer_config(fixture, "generate", inf_a)
        cfg_b = _infer_config(fixture, "generate", inf_b)
        cfg_b.text_format = "order"
        for cfg, _d in ((cfg_a, inf_a), (cfg_b, inf_b)):
            run_inference(
                cfg,
                training_config=fixture["training_config"],
                model=build_tiny(ext_vocab_file).eval(),
                vocoder=FakeVocoder(),
            )
        wa, _ = _read_wav(inf_a / "valid/wav/sess_w00000_ch0.wav")
        wb, _ = _read_wav(inf_b / "valid/wav/sess_w00000_ch0.wav")
        assert (wa == wb).all()
        meta = json.loads((inf_a / "valid/meta/sess_w00000.json").read_text("utf-8"))
        assert meta["text_format"] == "order" and "layout" not in meta

    def test_layout_is_the_only_key_mode_t_adds(self, fixture, ext_vocab5_file):
        # `TestModeParity` pins `set(gt_meta) == set(gen_meta)` in Mode O
        # only.  Mode T makes that invariant conditional: generate gains
        # "layout", which gt can NEVER carry (timestamps is rejected outside
        # generate).  Pin the conditional form here so the exception is a
        # checked contract, not just a narrated one - all three arms share
        # the 5-token vocab, so `text_format` is the only variable.
        metas = {}
        for name, mode, text_format in (
            ("gt", "gt", None),
            ("order", "generate", "order"),
            ("timestamps", "generate", "timestamps"),
        ):
            inf_dir = fixture["tmp_path"] / f"infer_keys_{name}"
            cfg = _infer_config(fixture, mode, inf_dir)
            if text_format is not None:
                cfg.text_format = text_format
            needs_model = mode != "gt"
            run_inference(
                cfg,
                training_config=fixture["training_config_5"],
                model=build_tiny(ext_vocab5_file).eval() if needs_model else None,
                vocoder=FakeVocoder() if needs_model else None,
            )
            metas[name] = json.loads(
                (inf_dir / "valid/meta/sess_w00000.json").read_text("utf-8")
            )

        assert set(metas["gt"]) == set(metas["order"])  # the Mode O invariant
        assert "layout" not in metas["order"] and "layout" in metas["timestamps"]
        assert set(metas["timestamps"]) - {"layout"} == set(metas["order"])

    def test_timestamps_rejected_outside_generate(self, fixture):
        cfg = _infer_config(fixture, "gt", fixture["tmp_path"] / "x")
        cfg.text_format = "timestamps"
        with pytest.raises(ValueError, match="text_format"):
            run_inference(cfg, training_config=fixture["training_config"])

    def test_unfittable_window_degrades_to_order(
        self, fixture, ext_vocab5_file, monkeypatch
    ):
        # Force the fit predicate to fail; the window must still generate.
        monkeypatch.setattr(inference_mod, "timestamp_fits", lambda *a, **k: False)
        inf_dir = fixture["tmp_path"] / "infer_deg"
        stats = run_inference(
            self._cfg(fixture, inf_dir),
            training_config=fixture["training_config_5"],
            model=build_tiny(ext_vocab5_file).eval(),
            vocoder=FakeVocoder(),
        )
        assert stats["n_timestamp_degraded"] == stats["n_selected"] > 0
        meta = json.loads((inf_dir / "valid/meta/sess_w00000.json").read_text("utf-8"))
        assert meta["text_format"] == "order"


# --------------------------------------------------------------------------- #
# Determinism across modes + generate/gt/resynth layout parity
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
            str(p.relative_to(test_dir)) for p in test_dir.rglob("*") if p.is_file()
        )
        return files

    def test_layout_identical_across_modes(self, fixture, ext_vocab_file):
        gt_dir = self._run_mode(fixture, "gt", ext_vocab_file)
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        res_dir = self._run_mode(fixture, "resynth", ext_vocab_file)
        assert self._layout(gt_dir) == self._layout(gen_dir) == self._layout(res_dir)

    def test_meta_keys_and_prompt_identical_across_modes(self, fixture, ext_vocab_file):
        # This is the determinism-across-modes contract: the SAME prompt
        # turns must be picked for generate and gt on the same seed.
        gt_dir = self._run_mode(fixture, "gt", ext_vocab_file)
        gen_dir = self._run_mode(fixture, "generate", ext_vocab_file)
        gt_meta = json.loads((gt_dir / "meta/sess_w00000.json").read_text("utf-8"))
        gen_meta = json.loads((gen_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert set(gt_meta) == set(gen_meta)
        assert [set(c) for c in gt_meta["channels"]] == [
            set(c) for c in gen_meta["channels"]
        ]
        # Prompt selection and window turns are mode-invariant; only rtf and
        # audio differ.
        for key in ("prompt", "turns", "window_duration_sec"):
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
        assert stats == {
            "n_selected": 2,
            "n_skipped": 0,
            "n_timestamp_degraded": 0,
        }

        test_dir = inf_dir / "valid"
        scp = (test_dir / "meta.scp").read_text("utf-8").splitlines()
        assert scp == [
            "sess_w00000 meta/sess_w00000.json",
            "sess_w00001 meta/sess_w00001.json",
        ]
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert meta["mode"] == "gt"
        assert meta["prompt"]["total_frames"] == 468
        assert meta["channels"][1]["ref_text"] == "bead cab"


class TestSelection:
    def test_window_with_only_in_window_turns_is_skipped(self, solo_window_fixture):
        # Single window on its session: every channel's only pool turns are
        # its own in-window turns, so the non-window tier is empty for every
        # channel -> the window is skipped and counted, never relaxed.
        inf_dir = solo_window_fixture["tmp_path"] / "infer_skip"
        cfg = _infer_config(solo_window_fixture, "gt", inf_dir)
        stats = run_inference(
            cfg, training_config=solo_window_fixture["training_config"]
        )
        assert stats["n_selected"] == 0
        assert stats["n_skipped"] == 1
        assert (
            not (inf_dir / "valid" / "meta.scp").exists()
            or (inf_dir / "valid" / "meta.scp").read_text("utf-8").strip() == ""
        )


# --------------------------------------------------------------------------- #
# Frozen eval manifest (shareable, prompt-pinning selection)
# --------------------------------------------------------------------------- #
def _window_c() -> WindowRecord:
    """A third window on the same session, so every window has TWO
    non-window candidates per channel and the ladder pick is a real draw
    rather than a forced one - which is what makes pinning observable."""
    return WindowRecord(
        window_id="sess_w00002",
        session_id="sess",
        audio_relpath="original/sess_mixed.flac",
        num_channels=2,
        sample_rate=SRC_SR,
        t0=45.0,
        t1=53.0,
        turns=(
            Turn(0, "spk_a", "fed ace", 45.5, 48.0),  # 2.5s
            Turn(1, "spk_b", "gab deaf", 48.5, 51.0),  # 2.5s
        ),
    )


@pytest.fixture
def three_window_fixture(tmp_path):
    return _write_fixture_files(tmp_path, [_window_a(), _window_b(), _window_c()], 60.0)


def _manifest_lines(header: dict, rows: list) -> str:
    return "".join(
        json.dumps(obj, ensure_ascii=False) + "\n" for obj in [header, *rows]
    )


def _header(**over) -> dict:
    base = {
        "record_type": "header",
        "manifest_version": 1,
        "split": "valid",
        "source_manifest": "valid.jsonl",
        "source_manifest_md5": "0" * 32,
    }
    base.update(over)
    return base


def _row(window_id, session_id, t0, t1, prompts) -> dict:
    return {
        "record_type": "window",
        "window_id": window_id,
        "session_id": session_id,
        "t0": t0,
        "t1": t1,
        "prompts": [{"channel": c, "start": s, "end": e} for c, s, e in prompts],
    }


class TestEvalManifestSelection:
    def _run_with_manifest(self, fixture, text, name="infer_pinned"):
        path = fixture["tmp_path"] / "eval_manifest.jsonl"
        path.write_text(text, encoding="utf-8")
        inf_dir = fixture["tmp_path"] / name
        cfg = _infer_config(fixture, "gt", inf_dir)
        cfg.selection.manifest = str(path)
        stats = run_inference(cfg, training_config=fixture["training_config"])
        return inf_dir / "valid", stats

    def test_manifest_pins_which_windows_run(self, three_window_fixture):
        # Only window B is listed, so only window B runs - even though the
        # seeded selection would have taken all three.
        text = _manifest_lines(
            _header(),
            [_row("sess_w00001", "sess", 25.0, 33.0, [(0, 5.5, 8.0), (1, 8.5, 11.0)])],
        )
        test_dir, stats = self._run_with_manifest(three_window_fixture, text)
        assert stats["n_selected"] == 1
        assert stats["n_skipped"] == 0
        assert (test_dir / "meta.scp").read_text("utf-8").splitlines() == [
            "sess_w00001 meta/sess_w00001.json"
        ]

    def test_manifest_pins_the_prompt_spans(self, three_window_fixture):
        # Window A's ch0 prompt is pinned to window C's turn.  The ladder is
        # free to prefer window B's; the meta must show the pinned one.
        text = _manifest_lines(
            _header(),
            [
                _row(
                    "sess_w00000", "sess", 5.0, 13.0, [(0, 45.5, 48.0), (1, 48.5, 51.0)]
                )
            ],
        )
        test_dir, _ = self._run_with_manifest(three_window_fixture, text)
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert [t["text"] for t in meta["prompt"]["turns"]] == [
            "fed ace",
            "gab deaf",
        ]

    def test_unknown_window_id_is_an_error(self, three_window_fixture):
        text = _manifest_lines(
            _header(),
            [_row("sess_wNOPE", "sess", 5.0, 13.0, [(0, 45.5, 48.0), (1, 48.5, 51.0)])],
        )
        with pytest.raises(ValueError, match="sess_wNOPE"):
            self._run_with_manifest(three_window_fixture, text, "infer_bad_id")

    def test_duplicate_window_id_is_an_error(self, three_window_fixture):
        row = _row("sess_w00000", "sess", 5.0, 13.0, [(0, 45.5, 48.0), (1, 48.5, 51.0)])
        text = _manifest_lines(_header(), [row, row])
        with pytest.raises(ValueError, match="duplicate"):
            self._run_with_manifest(three_window_fixture, text, "infer_dup")

    def test_window_geometry_mismatch_is_an_error(self, three_window_fixture):
        # t1 echo disagrees with the split -> the manifest was built against
        # different data.  Must fail loudly, never resolve silently.
        text = _manifest_lines(
            _header(),
            [
                _row(
                    "sess_w00000", "sess", 5.0, 99.0, [(0, 45.5, 48.0), (1, 48.5, 51.0)]
                )
            ],
        )
        with pytest.raises(ValueError, match="t1"):
            self._run_with_manifest(three_window_fixture, text, "infer_geom")

    def test_prompt_span_absent_from_the_pool_is_an_error(self, three_window_fixture):
        text = _manifest_lines(
            _header(),
            [
                _row(
                    "sess_w00000", "sess", 5.0, 13.0, [(0, 45.5, 47.0), (1, 48.5, 51.0)]
                )
            ],
        )
        with pytest.raises(ValueError, match="no pool turn"):
            self._run_with_manifest(three_window_fixture, text, "infer_span")

    def test_in_window_prompt_span_is_rejected(self, three_window_fixture):
        # Leakage is never allowed, not even when a manifest asks for it.
        text = _manifest_lines(
            _header(),
            [_row("sess_w00000", "sess", 5.0, 13.0, [(0, 5.5, 8.0), (1, 48.5, 51.0)])],
        )
        with pytest.raises(ValueError, match="overlaps"):
            self._run_with_manifest(three_window_fixture, text, "infer_leak")

    def test_missing_channel_is_an_error(self, three_window_fixture):
        # Every speaker must have a voice reference in the prompt.
        text = _manifest_lines(
            _header(),
            [_row("sess_w00000", "sess", 5.0, 13.0, [(0, 45.5, 48.0)])],
        )
        with pytest.raises(ValueError, match="channel"):
            self._run_with_manifest(three_window_fixture, text, "infer_chan")


class TestEvalManifestRoundTrip:
    def test_generated_manifest_reproduces_the_seeded_run_byte_for_byte(
        self, three_window_fixture
    ):
        """The acceptance test: freezing the seeded selection into a manifest
        and replaying it must change nothing at all."""
        from egs3.conversational.tts.src.eval_manifest import (
            build_eval_manifest,
            write_eval_manifest,
        )

        fixture = three_window_fixture
        seeded_dir = fixture["tmp_path"] / "infer_seeded"
        cfg = _infer_config(fixture, "gt", seeded_dir)
        seeded_stats = run_inference(cfg, training_config=fixture["training_config"])

        header, rows = build_eval_manifest(
            _infer_config(fixture, "gt", seeded_dir),
            training_config=fixture["training_config"],
        )
        assert header["num_windows"] == seeded_stats["n_selected"]
        path = fixture["tmp_path"] / "frozen.jsonl"
        write_eval_manifest(path, header, rows)

        pinned_dir = fixture["tmp_path"] / "infer_replay"
        cfg2 = _infer_config(fixture, "gt", pinned_dir)
        cfg2.selection.manifest = str(path)
        pinned_stats = run_inference(cfg2, training_config=fixture["training_config"])

        assert pinned_stats == seeded_stats
        a, b = seeded_dir / "valid", pinned_dir / "valid"
        assert (a / "meta.scp").read_bytes() == (b / "meta.scp").read_bytes()
        metas = sorted(p.name for p in (a / "meta").glob("*.json"))
        assert metas == sorted(p.name for p in (b / "meta").glob("*.json"))
        for name in metas:
            assert (a / "meta" / name).read_bytes() == (
                b / "meta" / name
            ).read_bytes(), name

    def test_builder_records_skipped_windows(self, solo_window_fixture):
        from egs3.conversational.tts.src.eval_manifest import build_eval_manifest

        header, rows = build_eval_manifest(
            _infer_config(solo_window_fixture, "gt", solo_window_fixture["tmp_path"]),
            training_config=solo_window_fixture["training_config"],
        )
        assert rows == []
        assert header["num_windows"] == 0
        assert header["num_skipped"] == 1


class TestManifestSlicing:
    """Slices are how a long run is sharded, so they must partition the
    manifest exactly - a dropped or duplicated window silently changes the
    test set."""

    def _slice(self, n_rows, n_slices):
        from egs3.conversational.tts.local.make_eval_manifest import slice_rows

        return slice_rows(list(range(n_rows)), n_slices)

    def test_slices_partition_in_order(self):
        parts = self._slice(1065, 8)
        assert len(parts) == 8
        assert [x for p in parts for x in p] == list(range(1065))
        assert sorted(len(p) for p in parts) == [133] * 7 + [134]

    def test_single_slice_is_the_whole_manifest(self):
        assert self._slice(5, 1) == [[0, 1, 2, 3, 4]]

    def test_more_slices_than_windows_is_an_error(self):
        with pytest.raises(ValueError, match="exceeds"):
            self._slice(3, 4)

    def test_zero_slices_is_an_error(self):
        with pytest.raises(ValueError, match=">= 1"):
            self._slice(3, 0)
