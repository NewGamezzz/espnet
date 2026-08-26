"""Concatenated-baseline tests: the pure layout/text helpers, the dispatch
literal, and a full CPU run over the fixture test set.

The layout helpers carry most of the weight here.  A concatenative baseline's
whole claim is "turn t starts exactly where turn t-1 ended, and no channel
speaks outside its own turns" - if that is wrong the interaction metrics are
silently meaningless rather than obviously broken, so it is pinned directly
rather than inferred from a run.
"""

from __future__ import annotations

import json

import pytest
import torch

from egs3.conversational.tts.src.concat_baseline import (
    MODE,
    concat_timeline,
    plain_text_ids,
    turn_spans,
)
from egs3.conversational.tts.src import system as system_mod


class TestModeLiteral:
    def test_dispatch_literal_matches_the_module(self):
        # system.py duplicates the mode string so importing the module is not
        # a side effect of dispatching; the two must not drift.
        assert system_mod.BASELINE_MODE == MODE


class TestPlainTextIds:
    """No <turn>/<OTHER>: a baseline that saw the conversational tokens
    would not be a baseline."""

    def _vocab(self):
        return {c: i for i, c in enumerate("abcdefg ")}

    def test_reference_and_target_are_joined_with_one_space(self):
        v = self._vocab()
        got = plain_text_ids("ab", "cd", v)
        assert got.tolist() == [[v[c] for c in "ab cd"]]

    def test_shape_is_single_branch(self):
        got = plain_text_ids("ab", "cd", self._vocab())
        assert got.shape[0] == 1

    def test_empty_reference_drops_the_separator(self):
        v = self._vocab()
        assert plain_text_ids("", "cd", v).tolist() == [[v["c"], v["d"]]]

    def test_unknown_character_raises(self):
        with pytest.raises(KeyError):
            plain_text_ids("ab", "zz", self._vocab())


class TestConcatTimeline:
    def test_turns_are_back_to_back_on_their_own_channels(self):
        a = torch.full((3,), 1.0)  # channel 0
        b = torch.full((2,), 2.0)  # channel 1
        c = torch.full((4,), 3.0)  # channel 0
        out = concat_timeline([a, b, c], [0, 1, 0], 2)
        assert out.shape == (2, 9)
        assert out[0].tolist() == [1, 1, 1, 0, 0, 3, 3, 3, 3]
        assert out[1].tolist() == [0, 0, 0, 2, 2, 0, 0, 0, 0]

    def test_no_two_channels_are_ever_active_at_once(self):
        # The defining property: zero overlap BY CONSTRUCTION.
        out = concat_timeline(
            [torch.ones(2), torch.ones(3), torch.ones(1)], [1, 0, 1], 2
        )
        active = (out != 0).sum(dim=0)
        assert active.max().item() == 1

    def test_total_length_is_the_sum_of_turn_lengths(self):
        waves = [torch.ones(n) for n in (5, 7, 2)]
        out = concat_timeline(waves, [0, 1, 0], 2)
        assert out.shape[1] == 14

    def test_two_dimensional_turn_waves_are_flattened(self):
        out = concat_timeline([torch.ones(1, 3), torch.ones(1, 2)], [0, 1], 2)
        assert out.shape == (2, 5)

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            concat_timeline([torch.ones(2)], [2], 2)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="turn channels"):
            concat_timeline([torch.ones(2), torch.ones(2)], [0], 2)

    def test_no_turns_raises(self):
        with pytest.raises(ValueError, match="no turns"):
            concat_timeline([], [], 2)


class TestTurnSpans:
    def test_spans_are_contiguous_in_output_time(self):
        spans = turn_spans(
            [torch.ones(24000), torch.ones(12000)], [0, 1], ["hi", "yo"], 24000
        )
        assert spans[0] == {
            "channel": 0,
            "text": "hi",
            "start": 0.0,
            "end": 1.0,
        }
        assert spans[1]["start"] == 1.0
        assert spans[1]["end"] == 1.5

    def test_every_turn_is_reported(self):
        spans = turn_spans([torch.ones(10)] * 4, [0, 1, 0, 1], list("abcd"), 24000)
        assert [s["text"] for s in spans] == ["a", "b", "c", "d"]
        # Contiguity: each turn starts where the previous one ended.
        for prev, nxt in zip(spans, spans[1:]):
            assert nxt["start"] == prev["end"]


# --------------------------------------------------------------------------- #
# Full CPU run over the fixture CoVoMix2 test set
# --------------------------------------------------------------------------- #
from omegaconf import OmegaConf  # noqa: E402

from egs3.conversational.tts.src.concat_baseline import (  # noqa: E402
    run_concat_baseline,
)
from .test_build_model import build_tiny  # noqa: E402
from .test_external_testset import _external_config, _write_testset  # noqa: E402
from .test_inference import FS, FakeVocoder, _read_wav  # noqa: E402


@pytest.fixture
def testset(tmp_path):
    return _write_testset(tmp_path)


@pytest.fixture
def tiny_model(testset):
    return build_tiny(testset["vocab"])


def _baseline_config(testset, inference_dir, **overrides):
    cfg = _external_config(testset, inference_dir, **overrides)
    cfg.mode = MODE
    cfg.source = "covomix2"
    return cfg


class TestConcatBaselineRun:
    def _run(self, testset, tiny_model, inference_dir, **overrides):
        cfg = _baseline_config(testset, inference_dir, **overrides)
        stats = run_concat_baseline(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        return inference_dir / "valid", stats, cfg

    def test_output_contract(self, testset, tiny_model, tmp_path):
        # Fixture: dialogue "000" has 3 turns, "001" has 2 -> 5 ODE calls,
        # one per turn, because the baseline never batches turns together.
        test_dir, stats, _ = self._run(testset, tiny_model, tmp_path / "infer")
        assert stats["n_selected"] == 2
        assert stats["n_turns"] == 5
        assert (test_dir / "meta.scp").read_text("utf-8").splitlines() == [
            "000 meta/000.json",
            "001 meta/001.json",
        ]
        for name in ("wav.scp", "prompt.scp", "text.scp", "mix.scp"):
            assert (test_dir / name).is_file()

        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["mode"] == MODE
        assert meta["turn_times"] == "concatenated"
        assert meta["baseline"]["prompt_policy"] == "fixed per-speaker reference"
        assert meta["baseline"]["layout"] == "back-to-back, zero gap"
        assert meta["baseline"]["n_turns"] == 3
        assert len(meta["turns"]) == 3
        assert len(meta["channels"]) == 2

    def test_turn_spans_tile_the_output_without_gaps(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer_spans")
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        spans = meta["turns"]
        assert spans[0]["start"] == 0.0
        for prev, nxt in zip(spans, spans[1:]):
            assert nxt["start"] == prev["end"]
        assert spans[-1]["end"] == pytest.approx(meta["window_duration_sec"], abs=1e-6)

    def test_channels_are_never_simultaneously_active(
        self, testset, tiny_model, tmp_path
    ):
        # The structural claim of the whole baseline, checked on real output.
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer_excl")
        ch0, sr = _read_wav(test_dir / "wav/000_ch0.wav")
        ch1, _ = _read_wav(test_dir / "wav/000_ch1.wav")
        assert sr == FS
        assert ch0.shape == ch1.shape
        both = (ch0 != 0) & (ch1 != 0)
        assert not both.any(), "a concatenated baseline must never overlap"

    def test_sharding_partitions_the_dialogues(self, testset, tiny_model, tmp_path):
        seen = []
        for i in range(2):
            test_dir, stats, _ = self._run(
                testset,
                tiny_model,
                tmp_path / f"infer_shard{i}",
                selection={"shard_index": i, "shard_count": 2},
            )
            scp = test_dir / f"meta.scp.{i}of2"
            assert scp.is_file(), "sharded runs must write suffixed SCPs"
            seen += [ln.split()[0] for ln in scp.read_text("utf-8").splitlines() if ln]
        assert sorted(seen) == ["000", "001"]

    def test_unknown_source_raises(self, testset, tiny_model, tmp_path):
        cfg = _baseline_config(testset, tmp_path / "infer_src")
        cfg.source = "nope"
        with pytest.raises(ValueError, match="unknown source"):
            run_concat_baseline(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_sssd_source_demands_a_frozen_manifest(self, testset, tiny_model, tmp_path):
        # Without pinned prompts the baseline would draw its own references
        # and the comparison would not be controlled.
        cfg = _baseline_config(testset, tmp_path / "infer_sssd")
        cfg.source = "sssd"
        cfg.dataset = OmegaConf.create({"split": "valid"})
        cfg.selection = OmegaConf.create({"manifest": None})
        with pytest.raises(ValueError, match="selection.manifest"):
            run_concat_baseline(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )
