"""Per-channel classifier-free guidance.

Why: on ZipVoice-Dialog test-en, channels whose whole script is a few
backchannels (<= 40 chars) emit a loud voiced DRONE instead of silence in
~1/3 of cases at cfg 3.0; cfg 2.0 removes it (loud-non-speech share 0.225
-> 0.004) but costs the talkative channel ~40 % relative WER.  Guidance is
therefore set PER CHANNEL from the script length of the current call.

Three layers, each pinned here: the upstream combine helper accepts a
per-row tensor (scalar path untouched), ``generate_batch`` builds that
tensor from ``GenerationItem.cfg_per_channel``, and the chunked driver
derives it from ``sampling.cfg_sparse_strength`` / ``cfg_sparse_max_chars``
(default null = bit-identical to before).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from egs3.conversational.tts.src.chunked_inference import run_chunked_inference
from egs3.conversational.tts.src.generation import (
    GenerationItem,
    generate_batch,
    pad_branch_text,
)
from espnet2.tts.f5.cfm import apply_cfg

from .conftest import randomize_params
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)
from .test_external_manifest import (
    ONE_SPK,
    TWO_SPK,
    _manifest_config,
    _meta,
    write_manifest,
)
from .test_inference import HOP, FakeVocoder, _read_wav


# --------------------------------------------------------------------------- #
# upstream combine helper
# --------------------------------------------------------------------------- #
class TestApplyCfg:
    def test_scalar_matches_the_original_formula(self):
        pred = torch.randn(3, 5, 4)
        null = torch.randn(3, 5, 4)
        assert torch.equal(apply_cfg(pred, null, 2.0), pred + (pred - null) * 2.0)

    def test_equal_tensor_is_bit_identical_to_scalar(self):
        pred = torch.randn(3, 5, 4)
        null = torch.randn(3, 5, 4)
        assert torch.equal(
            apply_cfg(pred, null, torch.full((3,), 2.0)), apply_cfg(pred, null, 2.0)
        )

    def test_per_row_tensor_applies_row_wise(self):
        pred = torch.randn(3, 5, 4)
        null = torch.randn(3, 5, 4)
        out = apply_cfg(pred, null, torch.tensor([3.0, 0.0, 2.0]))
        assert torch.equal(out[0], apply_cfg(pred[0:1], null[0:1], 3.0)[0])
        assert torch.equal(out[1], pred[1])  # zero guidance = conditional pred
        assert torch.equal(out[2], apply_cfg(pred[2:3], null[2:3], 2.0)[0])

    def test_wrong_tensor_length_raises(self):
        pred = torch.randn(3, 5, 4)
        with pytest.raises(ValueError, match="one guidance value per row"):
            apply_cfg(pred, pred, torch.tensor([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# generate_batch: per-item, per-channel guidance
# --------------------------------------------------------------------------- #
def _items(fx, model, cfgs):
    """Two dialogues (2 + 1 channels) as GenerationItems with optional
    per-channel guidance tuples."""
    from egs3.conversational.tts.src.external_testset import load_external_manifest
    from egs3.conversational.tts.src.generation import build_preprocessor

    records = load_external_manifest(fx["manifest"], fx["vocab"])
    pre = build_preprocessor(fx["training_config"])
    items = []
    for record, cfg in zip(records, cfgs):
        n = record.num_channels
        sample = pre(
            record.dialogue_id,
            {
                "turns": list(record.turns),
                "num_channels": n,
            },
        )
        text = pad_branch_text(sample, torch.device("cpu"))
        speech = torch.zeros(n, 24 * HOP)
        items.append(
            GenerationItem(
                speech=speech,
                text=text,
                prompt_frames=8,
                total_frames=24,
                cfg_per_channel=cfg,
            )
        )
    return items


def _run(model, items, cfg_strength):
    wavs, _ = generate_batch(
        model,
        FakeVocoder(),
        items,
        steps=2,
        cfg_strength=cfg_strength,
        sway_sampling_coef=-1.0,
        seed=0,
    )
    return wavs


class TestGenerateBatchPerChannel:
    def test_default_item_has_no_per_channel_guidance(self):
        item = GenerationItem(
            speech=torch.zeros(1, 8),
            text=torch.zeros(1, 2, dtype=torch.long),
            prompt_frames=1,
            total_frames=2,
        )
        assert item.cfg_per_channel is None

    def test_equal_per_channel_values_are_bit_identical_to_scalar(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        model = build_tiny(fx["vocab"])
        plain = _run(model, _items(fx, model, [None, None]), 2.0)
        tagged = _run(model, _items(fx, model, [(2.0, 2.0), (2.0,)]), 2.0)
        for a, b in zip(plain, tagged):
            assert torch.equal(a, b)

    def test_different_per_channel_values_change_the_output(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        # build_tiny's DiT init zeroes the conditioning paths, so CFG is
        # inert there (cfg 3.0 == cfg 0.0 bit-for-bit); re-randomize to
        # give guidance something to amplify.
        model = build_tiny(fx["vocab"])
        randomize_params(model, 0)
        plain = _run(model, _items(fx, model, [None, None]), 3.0)
        mixed = _run(model, _items(fx, model, [(3.0, 0.5), None]), 3.0)
        assert not torch.equal(plain[0], mixed[0])
        # the item WITHOUT a per-channel tuple keeps the scalar; only the
        # exchange-coupled rows of the tagged item move
        assert torch.equal(plain[1], mixed[1])

    def test_per_channel_length_must_match_the_channel_count(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        model = build_tiny(fx["vocab"])
        with pytest.raises(ValueError, match="cfg_per_channel"):
            _run(model, _items(fx, model, [(2.0,), None]), 3.0)


# --------------------------------------------------------------------------- #
# chunked driver: sparse-channel rule from the script length of the call
# --------------------------------------------------------------------------- #
def _run_chunked(fx, inference_dir, chunk, sampling_extra, *, randomize=False):
    cfg = _manifest_config(fx, inference_dir, chunk)
    cfg.sampling = OmegaConf.merge(cfg.sampling, sampling_extra)
    model = build_tiny(fx["vocab"])
    if randomize:  # see TestGenerateBatchPerChannel: CFG is inert on the raw tiny init
        randomize_params(model, 0)
    run_chunked_inference(
        cfg,
        training_config=fx["training_config"],
        model=model,
        vocoder=FakeVocoder(),
    )
    return inference_dir / "valid"


class TestSparseChannelGuidance:
    # TWO_SPK: ch0 "abc"+"gab" = 6 chars, ch1 "def" = 3 chars; ONE_SPK: 6 chars.
    def test_off_by_default_records_nothing_and_is_bit_identical(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        a = _run_chunked(fx, tmp_path / "a", {"turns": 10}, {})
        b = _run_chunked(
            fx, tmp_path / "b", {"turns": 10}, {"cfg_sparse_strength": None}
        )
        for wid in ("d2", "d1"):
            assert "cfg_per_channel" not in _meta(a, wid)["chunking"]["chunks"][0]
            assert "cfg_sparse_strength" not in _meta(a, wid)["chunking"]
            for ch in range(_meta(a, wid)["num_channels"]):
                x, _ = _read_wav(a / f"wav/{wid}_ch{ch}.wav")
                y, _ = _read_wav(b / f"wav/{wid}_ch{ch}.wav")
                assert (x == y).all()

    def test_sparse_channels_get_the_low_guidance(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        d = _run_chunked(
            fx,
            tmp_path / "s",
            {"turns": 10},
            {
                "cfg_strength": 3.0,
                "cfg_sparse_strength": 2.0,
                "cfg_sparse_max_chars": 4,
            },
        )
        two, one = _meta(d, "d2"), _meta(d, "d1")
        assert two["chunking"]["chunks"][0]["cfg_per_channel"] == [3.0, 2.0]
        assert one["chunking"]["chunks"][0]["cfg_per_channel"] == [3.0]
        assert two["chunking"]["cfg_sparse_strength"] == 2.0
        assert two["chunking"]["cfg_sparse_max_chars"] == 4

    def test_rule_uses_the_current_call_script_not_the_dialogue(self, tmp_path):
        # turns=1: each call carries ONE turn, so in call k only that turn's
        # channel has text; the other channel is sparse (0 chars) there.
        fx = write_manifest(tmp_path, [TWO_SPK])
        d = _run_chunked(
            fx,
            tmp_path / "s",
            {"turns": 1},
            {
                "cfg_strength": 3.0,
                "cfg_sparse_strength": 2.0,
                "cfg_sparse_max_chars": 0,
            },
        )
        chunks = _meta(d, "d2")["chunking"]["chunks"]
        # turns: (0,"abc"), (1,"def"), (0,"gab")
        assert [c["cfg_per_channel"] for c in chunks] == [
            [3.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
        ]

    def test_default_threshold_is_forty_chars(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        d = _run_chunked(
            fx, tmp_path / "s", {"turns": 10}, {"cfg_sparse_strength": 2.0}
        )
        m = _meta(d, "d2")
        assert m["chunking"]["cfg_sparse_max_chars"] == 40
        assert m["chunking"]["chunks"][0]["cfg_per_channel"] == [2.0, 2.0]

    def test_output_differs_from_the_global_setting(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        g = _run_chunked(
            fx, tmp_path / "g", {"turns": 10}, {"cfg_strength": 3.0}, randomize=True
        )
        s = _run_chunked(
            fx,
            tmp_path / "s",
            {"turns": 10},
            {
                "cfg_strength": 3.0,
                "cfg_sparse_strength": 0.5,
                "cfg_sparse_max_chars": 4,
            },
            randomize=True,
        )
        x, _ = _read_wav(g / "wav/d2_ch1.wav")
        y, _ = _read_wav(s / "wav/d2_ch1.wav")
        assert not (x == y).all()

    def test_invalid_values_raise(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        with pytest.raises(ValueError, match="cfg_sparse_strength"):
            _run_chunked(
                fx, tmp_path / "x", {"turns": 10}, {"cfg_sparse_strength": -1.0}
            )
        with pytest.raises(ValueError, match="cfg_sparse_max_chars"):
            _run_chunked(
                fx,
                tmp_path / "y",
                {"turns": 10},
                {"cfg_sparse_strength": 2.0, "cfg_sparse_max_chars": -3},
            )
