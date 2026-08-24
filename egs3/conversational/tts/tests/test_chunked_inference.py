"""Tests for the chunked external infer stage (src/chunked_inference.py).

Reuses the external-path fixtures: the fabricated CoVoMix2 tree, the tiny
random-init DiT, and FakeVocoder.  Duration expectations are written out
longhand (explicit per-speaker rates) so a shared bug in the formula under
test cannot pass both sides - same doctrine as test_external_testset.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from egs3.conversational.tts.dataset.preprocessing.text import (
    FRAMES_PER_SECOND,
    NEW_TOKENS,
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
    make_token2id,
)
from egs3.conversational.tts.src.chunked_inference import (
    MODE as CHUNKED_MODE,
    SPECIAL_TOKENS_PROMPT_FLOOR_SEC,
    CondComposition,
    SpecialTokensCond,
    TimestampText,
    _validated_chunk_cfg,
    call_turns,
    crossfade_concat,
    estimate_turn_secs,
    min_truncated_prompt_frames,
    run_chunked_inference,
    split_turns,
    tail_frames,
)
from egs3.conversational.tts.src.external_inference import (
    _prompt_turns,
    run_external_inference,
)
from egs3.conversational.tts.src.external_testset import (
    estimate_duration_sec,
    load_covomix2_testset,
)

from espnet3.systems.base.metric import measure

from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)
from .test_e2e_eval import (
    ASR_SUMMARY_KEYS,
    INTERACTION_SUMMARY_KEYS,
    QUALITY_SUMMARY_KEYS,
    SPEAKER_SUMMARY_KEYS,
)
from .test_external_testset import (
    DIALOGUES_3SPK,
    _external_config,
    _external_metrics_config,
    _write_testset,
)
from .test_inference import FS, HOP, FakeVocoder, _read_wav


@pytest.fixture
def testset(tmp_path):
    return _write_testset(tmp_path)


@pytest.fixture
def tiny_model(testset):
    return build_tiny(testset["vocab"])


def _records(testset):
    return load_covomix2_testset(
        testset["testset_root"], testset["librispeech_root"], testset["vocab"]
    )


# --------------------------------------------------------------------------- #
# split_turns (pure)
# --------------------------------------------------------------------------- #
class TestSplitTurns:
    def test_fixed_turn_count_partitions_with_remainder(self):
        assert split_turns([1.0] * 5, turns=2) == [(0, 2), (2, 4), (4, 5)]

    def test_turn_count_one_is_per_turn(self):
        assert split_turns([1.0, 2.0, 3.0], turns=1) == [(0, 1), (1, 2), (2, 3)]

    def test_target_sec_packs_greedily(self):
        # 10+10 <= 25, adding the third would exceed -> new chunk.
        assert split_turns([10.0, 10.0, 10.0], target_sec=25.0) == [(0, 2), (2, 3)]

    def test_oversized_single_turn_still_gets_own_chunk(self):
        # A 30 s turn exceeds the 25 s target on its own: never split inside
        # a turn, never merged with a neighbour.
        assert split_turns([30.0, 5.0], target_sec=25.0) == [(0, 1), (1, 2)]

    def test_ranges_partition_the_turns(self):
        for kwargs in ({"turns": 2}, {"target_sec": 3.5}):
            ranges = split_turns([1.0, 2.0, 3.0, 0.5, 4.0], **kwargs)
            flat = [i for a, b in ranges for i in range(a, b)]
            assert flat == list(range(5))

    def test_exactly_one_policy_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            split_turns([1.0])
        with pytest.raises(ValueError, match="exactly one"):
            split_turns([1.0], turns=2, target_sec=25.0)

    def test_bad_values_raise(self):
        with pytest.raises(ValueError):
            split_turns([1.0], turns=0)
        with pytest.raises(ValueError):
            split_turns([1.0], target_sec=0.0)
        with pytest.raises(ValueError, match="no turns"):
            split_turns([], turns=2)


class TestSplitTurnsCoverage:
    # 3 channels, strict round-robin, every turn 10 s, target 25 s.
    SECS = [10.0] * 6
    CHANNELS = [0, 1, 2, 0, 1, 2]

    def test_plain_greedy_would_close_before_channel_two_appears(self):
        assert split_turns(self.SECS, target_sec=25.0) == [(0, 2), (2, 4), (4, 6)]

    def test_chunk_may_not_close_until_every_channel_is_seen(self):
        got = split_turns(
            self.SECS,
            target_sec=25.0,
            channels=self.CHANNELS,
            num_channels=3,
            cover_all_speakers=True,
        )
        assert got == [(0, 3), (3, 6)]
        for a, b in got:
            assert set(self.CHANNELS[a:b]) == {0, 1, 2}

    def test_final_chunk_is_exempt_from_coverage(self):
        # 5 turns: the tail chunk (3, 5) holds channels {0, 1} only, which is
        # fine - the last chunk never conditions anything.
        got = split_turns(
            [10.0] * 5,
            target_sec=25.0,
            channels=[0, 1, 2, 0, 1],
            num_channels=3,
            cover_all_speakers=True,
        )
        assert got == [(0, 3), (3, 5)]
        assert set([0, 1, 2, 0, 1][3:5]) == {0, 1}

    def test_flag_off_ignores_channel_arguments(self):
        with_args = split_turns(
            self.SECS, target_sec=25.0, channels=self.CHANNELS, num_channels=3
        )
        assert with_args == split_turns(self.SECS, target_sec=25.0)

    def test_coverage_with_turns_policy_raises(self):
        with pytest.raises(ValueError, match="target_sec"):
            split_turns(
                self.SECS,
                turns=2,
                channels=self.CHANNELS,
                num_channels=3,
                cover_all_speakers=True,
            )

    def test_coverage_without_channels_raises(self):
        with pytest.raises(ValueError, match="channels"):
            split_turns(self.SECS, target_sec=25.0, cover_all_speakers=True)

    def test_channel_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="channels"):
            split_turns(
                self.SECS,
                target_sec=25.0,
                channels=[0, 1],
                num_channels=3,
                cover_all_speakers=True,
            )


# --------------------------------------------------------------------------- #
# estimate_turn_secs (pure)
# --------------------------------------------------------------------------- #
class TestEstimateTurnSecs:
    def test_per_turn_secs_use_the_speakers_own_rate(self, testset):
        record = _records(testset)[0]  # "000": 3 turns, channels 0, 1, 0
        prompt_secs = [1.0, 2.0]
        # Longhand: rate_ch = prompt_sec / prompt_chars (utf-8 bytes).
        rate0 = 1.0 / len(record.prompts[0].text.encode("utf-8"))
        rate1 = 2.0 / len(record.prompts[1].text.encode("utf-8"))
        got = estimate_turn_secs(record, prompt_secs, duration_scale=1.0, speed=1.0)
        expected = [
            len(record.turns[0].text.encode("utf-8")) * rate0,
            len(record.turns[1].text.encode("utf-8")) * rate1,
            len(record.turns[2].text.encode("utf-8")) * rate0,
        ]
        assert got == pytest.approx(expected)

    def test_sums_to_the_whole_dialogue_estimate(self, testset):
        for record in _records(testset):
            prompt_secs = [1.5, 0.75]
            per_turn = estimate_turn_secs(
                record, prompt_secs, duration_scale=1.117, speed=1.2
            )
            whole = estimate_duration_sec(
                record, prompt_secs, duration_scale=1.117, speed=1.2
            )
            assert sum(per_turn) == pytest.approx(whole)

    def test_wrong_prompt_count_raises(self, testset):
        record = _records(testset)[0]
        with pytest.raises(ValueError, match="prompt durations"):
            estimate_turn_secs(record, [1.0], duration_scale=1.0, speed=1.0)

    def test_degenerate_prompt_raises(self, testset):
        record = _records(testset)[0]
        with pytest.raises(ValueError, match="degenerate"):
            estimate_turn_secs(record, [0.0, 1.0], duration_scale=1.0, speed=1.0)

    def test_bad_speed_raises(self, testset):
        record = _records(testset)[0]
        with pytest.raises(ValueError, match="speed"):
            estimate_turn_secs(record, [1.0, 1.0], duration_scale=1.0, speed=0.0)


# --------------------------------------------------------------------------- #
# call_turns (pure)
# --------------------------------------------------------------------------- #
class TestCallTurns:
    def test_first_call_is_prompt_turns_plus_first_chunk(self, testset):
        record = _records(testset)[0]
        ranges = [(0, 2), (2, 3)]
        got = call_turns(record, ranges, 0)
        assert got[: len(record.prompts)] == _prompt_turns(record)
        assert got[len(record.prompts) :] == list(record.turns[0:2])

    def test_later_call_is_previous_chunk_plus_current(self, testset):
        record = _records(testset)[0]
        ranges = [(0, 2), (2, 3)]
        assert call_turns(record, ranges, 1) == list(record.turns[0:3])

    def test_token_budget_matches_the_masking_scheme(self, testset):
        # The <turn>/<OTHER> budget of call k's branch text, hand-computed:
        # branch i gets one <turn> per turn, then the turn's characters if it
        # owns the turn, else one <OTHER> per character.
        record = _records(testset)[0]
        ranges = [(0, 2), (2, 3)]
        turns = call_turns(record, ranges, 1)
        branches = build_branch_texts(turns, record.num_channels)
        for ch, branch in enumerate(branches):
            assert branch.count(TURN_TOKEN) == len(turns)
            expected_other = sum(len(t.text) for t in turns if t.channel != ch)
            assert branch.count(OTHER_TOKEN) == expected_other
            own = [tok for tok in branch if tok not in (TURN_TOKEN, OTHER_TOKEN)]
            assert len(own) == sum(len(t.text) for t in turns if t.channel == ch)

    def test_include_prompt_prepends_prompt_turns_after_round_zero(self, testset):
        record = _records(testset)[0]  # "000", 3 turns
        ranges = split_turns([1.0, 1.0, 1.0], turns=1)  # [(0,1),(1,2),(2,3)]
        got = call_turns(record, ranges, 1, include_prompt=True, history_chunks=1)
        assert got == _prompt_turns(record) + list(record.turns[0:2])

    def test_zero_history_is_prompt_plus_current_chunk_only(self, testset):
        record = _records(testset)[0]
        ranges = split_turns([1.0, 1.0, 1.0], turns=1)
        got = call_turns(record, ranges, 1, include_prompt=True, history_chunks=0)
        assert got == _prompt_turns(record) + list(record.turns[1:2])

    def test_all_history_spans_from_turn_zero(self, testset):
        record = _records(testset)[0]
        ranges = split_turns([1.0, 1.0, 1.0], turns=1)
        got = call_turns(record, ranges, 2, include_prompt=True, history_chunks=-1)
        assert got == _prompt_turns(record) + list(record.turns[0:3])

    def test_history_clamps_to_available_chunks(self, testset):
        record = _records(testset)[0]
        ranges = split_turns([1.0, 1.0, 1.0], turns=1)
        assert call_turns(
            record, ranges, 1, include_prompt=False, history_chunks=5
        ) == call_turns(record, ranges, 1, include_prompt=False, history_chunks=1)

    def test_defaults_reproduce_previous_chunk_behavior(self, testset):
        record = _records(testset)[0]
        ranges = split_turns([1.0, 1.0, 1.0], turns=2)  # [(0,2),(2,3)]
        assert call_turns(record, ranges, 1) == list(record.turns[0:3])


# --------------------------------------------------------------------------- #
# End-to-end chunked infer stage
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# crossfade_concat (pure)
# --------------------------------------------------------------------------- #
class TestCrossfadeConcat:
    def test_zero_fade_is_plain_concat(self):
        torch.manual_seed(0)
        chunks = [torch.randn(2, 100), torch.randn(2, 80)]
        out = crossfade_concat(chunks, 0)
        assert torch.equal(out, torch.cat(chunks, dim=1))

    def test_output_shrinks_by_one_fade_per_seam(self):
        torch.manual_seed(1)
        chunks = [torch.randn(2, 100), torch.randn(2, 80), torch.randn(2, 60)]
        out = crossfade_concat(chunks, 10)
        assert out.shape == (2, 100 + 80 + 60 - 2 * 10)

    def test_seam_uses_equal_power_gains(self):
        # ones->zeros isolates the falling gain, zeros->ones the rising one;
        # cos/sin at midpoint samples keeps summed POWER constant across the
        # seam (two different noise draws are uncorrelated, so linear gains
        # would dip -3 dB at the middle of every seam).
        f = 8
        ones, zeros = torch.ones(1, 40), torch.zeros(1, 40)
        down = crossfade_concat([ones, zeros], f)[0, 40 - f : 40]
        up = crossfade_concat([zeros, ones], f)[0, 40 - f : 40]
        t = (torch.arange(f) + 0.5) / f
        assert torch.allclose(down, torch.cos(t * torch.pi / 2))
        assert torch.allclose(up, torch.sin(t * torch.pi / 2))
        assert torch.allclose(down**2 + up**2, torch.ones(f))

    def test_untouched_regions_are_bit_identical(self):
        torch.manual_seed(2)
        a, b = torch.randn(2, 100), torch.randn(2, 80)
        out = crossfade_concat([a, b], 10)
        assert torch.equal(out[:, :90], a[:, :90])
        assert torch.equal(out[:, 100:], b[:, 10:])

    def test_fade_clamps_to_the_shorter_chunk(self):
        torch.manual_seed(3)
        a, b = torch.randn(1, 4), torch.randn(1, 30)
        out = crossfade_concat([a, b], 100)
        assert out.shape == (1, 4 + 30 - 4)

    def test_single_chunk_passes_through(self):
        torch.manual_seed(4)
        a = torch.randn(2, 50)
        assert torch.equal(crossfade_concat([a], 10), a)

    def test_negative_fade_raises(self):
        with pytest.raises(ValueError, match="fade"):
            crossfade_concat([torch.ones(1, 4)], -1)


# --------------------------------------------------------------------------- #
# conditioning-composition knobs (pure config validation)
# --------------------------------------------------------------------------- #
def _chunk_only_cfg(chunk):
    return OmegaConf.create({"chunk": chunk})


class TestCondCompositionConfig:
    def test_defaults_are_previous_chunk_only(self):
        _, _, _, comp, _, _, _ = _validated_chunk_cfg(_chunk_only_cfg({"turns": 2}))
        assert comp == CondComposition(include_prompt=False, history_chunks=1)

    def test_explicit_values_are_parsed(self):
        _, _, _, comp, _, _, _ = _validated_chunk_cfg(
            _chunk_only_cfg(
                {"turns": 2, "cond_include_prompt": True, "cond_history_chunks": 0}
            )
        )
        assert comp == CondComposition(include_prompt=True, history_chunks=0)

    def test_all_history_is_minus_one(self):
        _, _, _, comp, _, _, _ = _validated_chunk_cfg(
            _chunk_only_cfg(
                {"turns": 2, "cond_include_prompt": True, "cond_history_chunks": -1}
            )
        )
        assert comp.history_chunks == -1

    def test_no_conditioning_at_all_is_rejected(self):
        with pytest.raises(ValueError, match="cond_include_prompt"):
            _validated_chunk_cfg(
                _chunk_only_cfg({"turns": 2, "cond_history_chunks": 0})
            )

    def test_history_below_minus_one_is_rejected(self):
        with pytest.raises(ValueError, match="cond_history_chunks"):
            _validated_chunk_cfg(
                _chunk_only_cfg({"turns": 2, "cond_history_chunks": -2})
            )


class TestSpecialTokensConfig:
    def test_absent_cond_format_is_transcripts_mode(self):
        _, _, _, _, sptok, _, _ = _validated_chunk_cfg(_chunk_only_cfg({"turns": 2}))
        assert sptok is None

    def test_special_tokens_defaults(self):
        _, _, _, _, sptok, _, _ = _validated_chunk_cfg(
            _chunk_only_cfg({"target_sec": 25, "cond_format": "special_tokens"})
        )
        assert sptok == SpecialTokensCond(prompt_sec=8.0, prev_sec=10.0)

    def test_knobs_are_parsed_and_zero_prev_is_reanchor(self):
        _, _, _, _, sptok, _, _ = _validated_chunk_cfg(
            _chunk_only_cfg(
                {
                    "target_sec": 25,
                    "cond_format": "special_tokens",
                    "cond_prompt_sec": 4.0,
                    "cond_prev_sec": 0.0,
                }
            )
        )
        assert sptok == SpecialTokensCond(prompt_sec=4.0, prev_sec=0.0)

    def test_knobs_require_special_tokens_mode(self):
        with pytest.raises(ValueError, match="cond_prompt_sec"):
            _validated_chunk_cfg(_chunk_only_cfg({"turns": 2, "cond_prompt_sec": 4.0}))
        with pytest.raises(ValueError, match="cond_prev_sec"):
            _validated_chunk_cfg(_chunk_only_cfg({"turns": 2, "cond_prev_sec": 4.0}))

    def test_transcript_composition_knobs_rejected_in_special_mode(self):
        with pytest.raises(ValueError, match="cond_include_prompt"):
            _validated_chunk_cfg(
                _chunk_only_cfg(
                    {
                        "turns": 2,
                        "cond_format": "special_tokens",
                        "cond_include_prompt": True,
                    }
                )
            )
        with pytest.raises(ValueError, match="cond_history_chunks"):
            _validated_chunk_cfg(
                _chunk_only_cfg(
                    {
                        "turns": 2,
                        "cond_format": "special_tokens",
                        "cond_history_chunks": 2,
                    }
                )
            )

    def test_bad_values_rejected(self):
        with pytest.raises(ValueError, match="cond_format"):
            _validated_chunk_cfg(_chunk_only_cfg({"turns": 2, "cond_format": "sp"}))
        with pytest.raises(ValueError, match="cond_prompt_sec"):
            _validated_chunk_cfg(
                _chunk_only_cfg(
                    {
                        "turns": 2,
                        "cond_format": "special_tokens",
                        "cond_prompt_sec": 0.0,
                    }
                )
            )
        with pytest.raises(ValueError, match="cond_prev_sec"):
            _validated_chunk_cfg(
                _chunk_only_cfg(
                    {"turns": 2, "cond_format": "special_tokens", "cond_prev_sec": -1.0}
                )
            )


class TestTimestampTextConfig:
    def test_absent_is_mode_o(self):
        *_, tsl = _validated_chunk_cfg(_chunk_only_cfg({"turns": 2}))
        assert tsl is None

    def test_explicit_order_is_mode_o(self):
        *_, tsl = _validated_chunk_cfg(
            _chunk_only_cfg({"turns": 2, "text_format": "order"})
        )
        assert tsl is None

    def test_timestamps_defaults(self):
        *_, tsl = _validated_chunk_cfg(_chunk_only_cfg(
            {"turns": 2, "cond_format": "special_tokens", "text_format": "timestamps"}))
        assert tsl == TimestampText(gap_sec=0.4)

    def test_gap_parsed(self):
        *_, tsl = _validated_chunk_cfg(_chunk_only_cfg(
            {"turns": 2, "cond_format": "special_tokens", "text_format": "timestamps",
             "turn_gap_sec": 0.0}))
        assert tsl == TimestampText(gap_sec=0.0)

    def test_requires_special_tokens(self):
        with pytest.raises(ValueError, match="cond_format: special_tokens"):
            _validated_chunk_cfg(
                _chunk_only_cfg({"turns": 2, "text_format": "timestamps"})
            )

    def test_gap_requires_timestamps(self):
        with pytest.raises(ValueError, match="turn_gap_sec"):
            _validated_chunk_cfg(_chunk_only_cfg(
                {"turns": 2, "cond_format": "special_tokens", "turn_gap_sec": 0.4}))

    def test_rejects_cross_fade(self):
        # A fade eats audio at every seam, so turn k would land
        # k * cross_fade_sec earlier than the layout the text was written
        # against - the timing metric would read the artifact as drift.
        with pytest.raises(ValueError, match="cross_fade_sec"):
            _validated_chunk_cfg(_chunk_only_cfg(
                {"turns": 2, "cond_format": "special_tokens",
                 "text_format": "timestamps", "cross_fade_sec": 0.1}))
        # ... but a fade stays legal in Mode O, and an explicit 0.0 is fine.
        *_, tsl = _validated_chunk_cfg(_chunk_only_cfg(
            {"turns": 2, "cond_format": "special_tokens",
             "text_format": "timestamps", "cross_fade_sec": 0.0}))
        assert tsl == TimestampText(gap_sec=0.4)

    def test_bad_values(self):
        with pytest.raises(ValueError, match="text_format"):
            _validated_chunk_cfg(_chunk_only_cfg({"turns": 2, "text_format": "modeT"}))
        with pytest.raises(ValueError, match="turn_gap_sec"):
            _validated_chunk_cfg(_chunk_only_cfg(
                {"turns": 2, "cond_format": "special_tokens",
                 "text_format": "timestamps", "turn_gap_sec": -1.0}))


class TestFrameHelpers:
    FS, HOP = 24000, 256

    def test_min_truncation_uses_shortest_prompt(self):
        # 4.0 s and 6.0 s prompts, cap 8.0 -> 4.0 s -> 96000 // 256 = 375.
        assert (
            min_truncated_prompt_frames(
                [4 * self.FS, 6 * self.FS], 8.0, self.FS, self.HOP
            )
            == 375
        )

    def test_cap_binds_when_prompts_are_long(self):
        # cap 3.0 s -> 72000 // 256 = 281 (floor of 281.25).
        assert (
            min_truncated_prompt_frames(
                [4 * self.FS, 6 * self.FS], 3.0, self.FS, self.HOP
            )
            == 281
        )

    def test_tail_frames_floor_and_clamp(self):
        # 5.0 s tail of ample audio -> 120000 // 256 = 468 (floor of 468.75).
        assert tail_frames(30 * self.FS, 5.0, self.FS, self.HOP) == 468
        # less generated audio than prev_sec -> whole tail, floored to hops.
        assert tail_frames(10000, 5.0, self.FS, self.HOP) == 10000 // self.HOP
        assert tail_frames(30 * self.FS, 0.0, self.FS, self.HOP) == 0


def _chunked_config(testset, inference_dir, chunk, **overrides):
    cfg = _external_config(testset, inference_dir, **overrides)
    cfg.mode = CHUNKED_MODE
    cfg.chunk = OmegaConf.create(chunk)
    return cfg


class TestChunkedInfer:
    def _run(self, testset, tiny_model, inference_dir, chunk, **overrides):
        cfg = _chunked_config(testset, inference_dir, chunk, **overrides)
        stats = run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        return inference_dir / "valid", stats, cfg

    def test_output_contract_and_chunking_meta(self, testset, tiny_model, tmp_path):
        # DIALOGUES: "000" has 3 turns -> 2 chunks at turns=2; "001" has 2
        # turns -> 1 chunk.  Two rounds total, three ODE calls.
        test_dir, stats, _ = self._run(
            testset, tiny_model, tmp_path / "infer", {"turns": 2}
        )
        assert stats == {
            "n_selected": 2,
            "n_skipped": 0,
            "n_not_sampled": 0,
            "n_other_shards": 0,
            "n_rounds": 2,
            "n_batches": 3,  # no batching block -> every call is a singleton
        }
        assert (test_dir / "meta.scp").read_text("utf-8").splitlines() == [
            "000 meta/000.json",
            "001 meta/001.json",
        ]
        for name in ("wav.scp", "prompt.scp", "text.scp", "mix.scp"):
            assert (test_dir / name).is_file()
        assert not (test_dir / "gt").exists()

        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["mode"] == CHUNKED_MODE
        assert meta["has_reference_audio"] is False
        assert meta["turn_times"] == "ordinal"
        assert meta["rtf"] is None
        chunking = meta["chunking"]
        assert chunking["policy"] == {"turns": 2}
        assert chunking["n_chunks"] == 2
        assert chunking["oversized"] == [False, False]
        assert [(c["turn_start"], c["turn_end"]) for c in chunking["chunks"]] == [
            (0, 2),
            (2, 3),
        ]
        assert [c["round"] for c in chunking["chunks"]] == [0, 1]
        for c in chunking["chunks"]:
            assert c["gen_frames"] >= 1
            assert c["batch_size"] == 1
            assert c["batch_elapsed_sec"] > 0

        meta1 = json.loads((test_dir / "meta/001.json").read_text("utf-8"))
        assert meta1["chunking"]["n_chunks"] == 1

    def test_wave_is_the_concat_of_chunk_regions(self, testset, tiny_model, tmp_path):
        test_dir, _, _ = self._run(
            testset, tiny_model, tmp_path / "infer", {"turns": 2}
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        total_frames = sum(c["gen_frames"] for c in meta["chunking"]["chunks"])
        wav, sr = _read_wav(test_dir / "wav/000_ch0.wav")
        assert sr == FS
        assert wav.shape[0] == total_frames * HOP
        assert meta["window_duration_sec"] == pytest.approx(total_frames * HOP / FS)
        mix, _ = _read_wav(test_dir / "mix/000.wav")
        assert mix.shape[0] == total_frames * HOP

    def test_target_sec_policy_flags_oversized_chunks(
        self, testset, tiny_model, tmp_path
    ):
        # A microscopic target makes every turn its own chunk and every
        # chunk oversized - exercises both the split and the flag.
        test_dir, stats, _ = self._run(
            testset, tiny_model, tmp_path / "infer", {"target_sec": 0.001}
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["chunking"]["policy"] == {"target_sec": 0.001}
        assert meta["chunking"]["n_chunks"] == 3
        assert meta["chunking"]["oversized"] == [True, True, True]
        assert stats["n_rounds"] == 3

    def test_cross_fade_shortens_output_and_is_recorded(
        self, testset, tiny_model, tmp_path
    ):
        # One hop of fade: <= every chunk (gen_frames >= 1), so no clamping
        # and the expected length is exact.  Dialogue "000" has 2 chunks at
        # turns=2 -> one seam; "001" has 1 chunk -> no seam.
        fade_sec = HOP / FS
        test_dir, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {"turns": 2, "cross_fade_sec": fade_sec},
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        chunking = meta["chunking"]
        assert chunking["policy"] == {"turns": 2}  # seam knob is not a policy
        assert chunking["cross_fade_sec"] == pytest.approx(fade_sec)
        total_frames = sum(c["gen_frames"] for c in chunking["chunks"])
        n_seams = chunking["n_chunks"] - 1
        expected = total_frames * HOP - n_seams * HOP
        for ch in range(2):
            wav, _ = _read_wav(test_dir / f"wav/000_ch{ch}.wav")
            assert wav.shape[0] == expected
        mix, _ = _read_wav(test_dir / "mix/000.wav")
        assert mix.shape[0] == expected
        assert meta["window_duration_sec"] == pytest.approx(expected / FS)

        meta1 = json.loads((test_dir / "meta/001.json").read_text("utf-8"))
        assert meta1["chunking"]["cross_fade_sec"] == pytest.approx(fade_sec)
        frames1 = sum(c["gen_frames"] for c in meta1["chunking"]["chunks"])
        wav1, _ = _read_wav(test_dir / "wav/001_ch0.wav")
        assert wav1.shape[0] == frames1 * HOP

    def test_zero_cross_fade_is_bit_identical_to_hard_concat(
        self, testset, tiny_model, tmp_path
    ):
        hard_dir, _, _ = self._run(testset, tiny_model, tmp_path / "hard", {"turns": 2})
        zero_dir, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "zero",
            {"turns": 2, "cross_fade_sec": 0.0},
        )
        meta = json.loads((hard_dir / "meta/000.json").read_text("utf-8"))
        assert meta["chunking"]["cross_fade_sec"] == 0.0
        for rel in ("wav/000_ch0.wav", "wav/000_ch1.wav", "mix/000.wav"):
            hard, _ = _read_wav(hard_dir / rel)
            zero, _ = _read_wav(zero_dir / rel)
            assert (hard == zero).all(), rel

    def test_negative_cross_fade_is_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset, tmp_path / "infer", {"turns": 2, "cross_fade_sec": -0.1}
        )
        with pytest.raises(ValueError, match="cross_fade_sec"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_missing_chunk_block_is_rejected(self, testset, tiny_model, tmp_path):
        cfg = _external_config(testset, tmp_path / "infer")
        cfg.mode = CHUNKED_MODE
        with pytest.raises(ValueError, match="chunk"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_wrong_mode_is_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(testset, tmp_path / "infer", {"turns": 2})
        cfg.mode = "generate_external"
        with pytest.raises(ValueError, match="expected mode"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_round_k_conditions_on_round_k_minus_one(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        # Capture every GenerationItem handed to generate_batch and check
        # the round-1 call's prompt equals a round-0 generated wave.
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            out = real(model, vocoder, items, **kwargs)
            captured.append((items, kwargs, out))
            return out

        monkeypatch.setattr(ci, "generate_batch", spy)
        self._run(testset, tiny_model, tmp_path / "infer", {"turns": 2})

        # Three singleton calls; the round-1 call is the last one.
        round1_items, _, _ = captured[-1]
        item = round1_items[0]
        for items, kwargs, out in captured[:-1]:
            wavs = out[0][0]
            frames = wavs.shape[1] // HOP
            if item.prompt_frames == frames:
                prompt = item.speech[:, : frames * HOP].cpu()
                if torch.equal(prompt, wavs[:, : frames * HOP]):
                    break
        else:
            pytest.fail("round-1 prompt does not match any round-0 output")
        # And the generated region is zeros.
        gen_region = item.speech[:, item.prompt_frames * HOP :]
        assert torch.equal(gen_region, torch.zeros_like(gen_region))

    def test_rounds_reseed_differently(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        import egs3.conversational.tts.src.chunked_inference as ci

        seeds = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            seeds.append(kwargs["seed"])
            return real(model, vocoder, items, **kwargs)

        monkeypatch.setattr(ci, "generate_batch", spy)
        self._run(testset, tiny_model, tmp_path / "infer", {"turns": 2})
        # Config seed is 0; round 0 calls use 0, the round-1 call uses 1.
        assert sorted(seeds) == [0, 0, 1]


# --------------------------------------------------------------------------- #
# conditioning composition end-to-end
# --------------------------------------------------------------------------- #
class TestCondCompositionInfer:
    def _spy_run(self, testset, tiny_model, inference_dir, chunk, **kwargs):
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kw):
            out = real(model, vocoder, items, **kw)
            captured.append((items[0], out[0][0]))
            return out

        monkeypatch = kwargs.pop("monkeypatch")
        monkeypatch.setattr(ci, "generate_batch", spy)
        cfg = _chunked_config(testset, inference_dir, chunk)
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
            **kwargs,
        )
        return captured

    @staticmethod
    def _prompt0(captured):
        item0, _ = captured[0]
        return item0.speech[:, : item0.prompt_frames * HOP].cpu()

    def test_reanchor_conditions_every_round_on_the_prompt(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        captured = self._spy_run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {"turns": 1, "cond_include_prompt": True, "cond_history_chunks": 0},
            monkeypatch=monkeypatch,
        )
        assert len(captured) == 5
        prompt0 = self._prompt0(captured)
        for item, _ in (captured[2], captured[4]):  # 000 rounds 1 and 2
            got = item.speech[:, : item.prompt_frames * HOP].cpu()
            assert torch.equal(got, prompt0)

    def test_prompt_plus_last_chunk_conditioning(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        captured = self._spy_run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {"turns": 1, "cond_include_prompt": True, "cond_history_chunks": 1},
            monkeypatch=monkeypatch,
        )
        prompt0 = self._prompt0(captured)
        chunk0 = captured[0][1]  # 000 round-0 output
        item, _ = captured[2]  # 000 round 1
        expected = torch.cat([prompt0, chunk0], dim=1)
        assert item.prompt_frames * HOP == expected.shape[1]
        assert torch.equal(item.speech[:, : expected.shape[1]].cpu(), expected)

    def test_all_history_conditioning_grows_with_depth(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        captured = self._spy_run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {"turns": 1, "cond_include_prompt": True, "cond_history_chunks": -1},
            monkeypatch=monkeypatch,
        )
        prompt0 = self._prompt0(captured)
        chunk0, chunk1 = captured[0][1], captured[2][1]  # 000 rounds 0, 1
        item, _ = captured[4]  # 000 round 2
        expected = torch.cat([prompt0, chunk0, chunk1], dim=1)
        assert item.prompt_frames * HOP == expected.shape[1]
        assert torch.equal(item.speech[:, : expected.shape[1]].cpu(), expected)

    def test_defaults_stay_bit_identical_to_previous_chunk_only(
        self, testset, tiny_model, tmp_path
    ):
        for name, chunk in (
            ("implicit", {"turns": 2}),
            (
                "explicit",
                {"turns": 2, "cond_include_prompt": False, "cond_history_chunks": 1},
            ),
        ):
            cfg = _chunked_config(testset, tmp_path / name, chunk)
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )
        for rel in ("wav/000_ch0.wav", "wav/000_ch1.wav", "mix/000.wav"):
            a, _ = _read_wav(tmp_path / "implicit/valid" / rel)
            b, _ = _read_wav(tmp_path / "explicit/valid" / rel)
            assert (a == b).all(), rel

    def test_meta_records_the_composition(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset,
            tmp_path / "infer",
            {"turns": 2, "cond_include_prompt": True, "cond_history_chunks": -1},
        )
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        meta = json.loads((tmp_path / "infer/valid/meta/000.json").read_text("utf-8"))
        assert meta["chunking"]["cond_include_prompt"] is True
        assert meta["chunking"]["cond_history_chunks"] == -1

    def test_hygiene_touches_history_but_never_the_prompt_segment(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        captured = self._spy_run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {
                "turns": 1,
                "cond_include_prompt": True,
                "cond_history_chunks": 1,
                "cond_silence_gate": True,
            },
            monkeypatch=monkeypatch,
            speech_regions_fn=lambda wav, fs, threshold: [],
        )
        prompt0 = self._prompt0(captured)
        chunk0 = captured[0][1]
        item, _ = captured[2]  # 000 round 1
        cut = prompt0.shape[1]
        assert torch.equal(item.speech[:, :cut].cpu(), prompt0)
        history = item.speech[:, cut : item.prompt_frames * HOP].cpu()
        assert history.shape[1] == chunk0.shape[1]
        assert not torch.equal(history, chunk0)  # fully gated to room tone

    def test_prompt_only_conditioning_skips_hygiene_entirely(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        calls = []
        self._spy_run(
            testset,
            tiny_model,
            tmp_path / "infer",
            {
                "turns": 1,
                "cond_include_prompt": True,
                "cond_history_chunks": 0,
                "cond_silence_gate": True,
            },
            monkeypatch=monkeypatch,
            speech_regions_fn=lambda wav, fs, threshold: calls.append(1) or [],
        )
        assert calls == []  # no generated segment -> the gate never runs
        meta = json.loads((tmp_path / "infer/valid/meta/000.json").read_text("utf-8"))
        for c in meta["chunking"]["chunks"]:
            assert "conditioning" not in c


# --------------------------------------------------------------------------- #
# 3-channel end-to-end: cover_all_speakers + the empty-channel guard
# --------------------------------------------------------------------------- #
class TestThreeSpeakerChunked:
    def _run3(self, tmp_path, chunk, dialogues=None, ids=None):
        ts = _write_testset(tmp_path, dialogues=dialogues or DIALOGUES_3SPK)
        model = build_tiny(ts["vocab"])
        cfg = _chunked_config(ts, tmp_path / "infer", chunk)
        cfg.testset.num_channels = 3
        if ids is not None:
            ids_file = tmp_path / "ids.txt"
            ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
            cfg.selection = OmegaConf.create({"dialogue_ids": str(ids_file)})
        counts = run_chunked_inference(
            cfg,
            training_config=ts["training_config"],
            model=model,
            vocoder=FakeVocoder(),
        )
        return ts, cfg, counts

    def test_three_channel_output_contract(self, tmp_path):
        ts, cfg, counts = self._run3(tmp_path, {"target_sec": 2.0})
        test_dir = Path(cfg.inference_dir) / cfg.test_name
        meta = json.loads((test_dir / "meta" / "900.json").read_text())
        assert meta["num_channels"] == 3
        assert len(meta["channels"]) == 3
        assert len(meta["prompt"]["turns"]) == 3
        for ch in range(3):
            assert (test_dir / "wav" / f"900_ch{ch}.wav").is_file()
            assert (test_dir / "prompt" / f"900_ch{ch}.wav").is_file()

    def test_cover_all_speakers_is_recorded_and_holds(self, tmp_path):
        # Tiny target so plain greedy WOULD close one-turn chunks; coverage
        # must hold anyway on every non-final chunk.
        ts, cfg, _ = self._run3(
            tmp_path, {"target_sec": 0.05, "cover_all_speakers": True}
        )
        test_dir = Path(cfg.inference_dir) / cfg.test_name
        oversized_flags = []
        for wid in ("900", "901"):
            meta = json.loads((test_dir / "meta" / f"{wid}.json").read_text())
            assert meta["chunking"]["cover_all_speakers"] is True
            chunks = meta["chunking"]["chunks"]
            turns = meta["turns"]
            oversized_flags.extend(meta["chunking"]["oversized"])
            for entry in chunks[:-1]:  # final chunk exempt
                chans = {
                    turns[i]["channel"]
                    for i in range(entry["turn_start"], entry["turn_end"])
                }
                assert chans == {0, 1, 2}
        # Coverage-forced overruns flag oversized: with target_sec=0.05
        # and 3-channel coverage, every non-degenerate chunk exceeds target.
        assert any(oversized_flags)

    def test_cover_all_speakers_with_turns_policy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="target_sec"):
            self._run3(tmp_path, {"turns": 2, "cover_all_speakers": True})

    def test_selected_dialogue_with_an_empty_channel_is_rejected(self, tmp_path):
        crippled = dict(DIALOGUES_3SPK)
        crippled["902"] = (
            ["abc def", "bead cab"],  # 2 turns -> channel 2 never speaks
            DIALOGUES_3SPK["900"][1],
        )
        with pytest.raises(ValueError, match="902.*no turns"):
            self._run3(tmp_path, {"target_sec": 2.0}, dialogues=crippled)

    def test_unselected_empty_channel_dialogue_does_not_break_the_run(self, tmp_path):
        crippled = dict(DIALOGUES_3SPK)
        crippled["902"] = (
            ["abc def", "bead cab"],
            DIALOGUES_3SPK["900"][1],
        )
        ts, cfg, counts = self._run3(
            tmp_path,
            {"target_sec": 2.0},
            dialogues=crippled,
            ids=["900", "901"],
        )
        assert counts["n_selected"] == 2


# --------------------------------------------------------------------------- #
# Reduction property and determinism
# --------------------------------------------------------------------------- #
class TestReductionAndDeterminism:
    def test_single_chunk_reduces_to_the_unchunked_path(
        self, testset, tiny_model, tmp_path
    ):
        # chunk.turns >= max turn count -> one call per dialogue whose
        # conditioning (prompt blocks + all turns' text) and round-0 seed
        # (base + 0) are identical to generate_external's -> bit-equal audio.
        ext_cfg = _external_config(testset, tmp_path / "ext")
        run_external_inference(
            ext_cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        chk_cfg = _chunked_config(testset, tmp_path / "chk", {"turns": 99})
        run_chunked_inference(
            chk_cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        for rel in (
            "wav/000_ch0.wav",
            "wav/000_ch1.wav",
            "wav/001_ch0.wav",
            "wav/001_ch1.wav",
            "mix/000.wav",
            "mix/001.wav",
        ):
            ext, _ = _read_wav(tmp_path / "ext/valid" / rel)
            chk, _ = _read_wav(tmp_path / "chk/valid" / rel)
            assert (ext == chk).all(), rel

    def test_rerunning_a_shard_is_bit_identical(self, testset, tiny_model, tmp_path):
        outs = []
        for run_dir in ("a", "b"):
            cfg = _chunked_config(
                testset,
                tmp_path / run_dir,
                {"turns": 2},
                selection={
                    "min_duration": None,
                    "max_duration": None,
                    "num_dialogues": None,
                    "seed": 0,
                    "shard_index": 0,
                    "shard_count": 2,
                },
            )
            stats = run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )
            outs.append((tmp_path / run_dir / "valid", stats))
        (dir_a, stats_a), (dir_b, stats_b) = outs
        assert stats_a == stats_b
        assert stats_a["n_other_shards"] == 1  # 2 dialogues, 2 shards
        scp_a = (dir_a / "wav.scp.0of2").read_text("utf-8")
        assert scp_a == (dir_b / "wav.scp.0of2").read_text("utf-8")
        rel = scp_a.splitlines()[0].split()[1]
        wav_a, _ = _read_wav(dir_a / rel)
        wav_b, _ = _read_wav(dir_b / rel)
        assert (wav_a == wav_b).all(), rel

    def test_shards_partition_the_dialogues(self, testset, tiny_model, tmp_path):
        seen = []
        for shard_index in (0, 1):
            cfg = _chunked_config(
                testset,
                tmp_path / f"s{shard_index}",
                {"turns": 2},
                selection={
                    "min_duration": None,
                    "max_duration": None,
                    "num_dialogues": None,
                    "seed": 0,
                    "shard_index": shard_index,
                    "shard_count": 2,
                },
            )
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )
            scp = tmp_path / f"s{shard_index}/valid/meta.scp.{shard_index}of2"
            seen += [ln.split()[0] for ln in scp.read_text("utf-8").splitlines()]
        assert sorted(seen) == ["000", "001"]


# --------------------------------------------------------------------------- #
# Conditioning hygiene: silence gate + loudness normalization
# --------------------------------------------------------------------------- #
def _rms_frames(wav_1d: torch.Tensor, fs: int) -> torch.Tensor:
    """Longhand 20 ms frame RMS, duplicated here so a shared bug in the
    production framing cannot pass both sides (same doctrine as the
    duration tests)."""
    frame = fs // 50
    n = wav_1d.shape[0] // frame
    return wav_1d[: n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt()


def _expected_active_rms(wav_1d: torch.Tensor, fs: int) -> float | None:
    rms = _rms_frames(wav_1d, fs)
    active = rms > 1e-3
    if not bool(active.any()):
        return None
    return float(rms[active].mean())


class TestActiveRms:
    def test_ignores_silent_frames(self):
        from egs3.conversational.tts.src.chunked_inference import active_rms

        wav = torch.zeros(FS)
        wav[: FS // 2] = 0.5
        assert active_rms(wav, FS) == pytest.approx(0.5, abs=1e-6)

    def test_all_silent_returns_none(self):
        from egs3.conversational.tts.src.chunked_inference import active_rms

        assert active_rms(torch.full((FS,), 1e-5), FS) is None


class TestSilenceGate:
    def test_zeros_outside_speech_regions(self):
        from egs3.conversational.tts.src.chunked_inference import silence_gate

        wav = torch.randn(2, 1000)
        calls = []

        def regions(ch_wav, fs, threshold):
            calls.append((ch_wav.shape, fs, threshold))
            return [(100, 400)] if len(calls) == 1 else []

        gated, frac = silence_gate(wav, FS, threshold=0.2, speech_regions_fn=regions)
        assert calls == [((1000,), FS, 0.2), ((1000,), FS, 0.2)]
        assert torch.equal(gated[0, 100:400], wav[0, 100:400])
        assert torch.equal(gated[0, :100], torch.zeros(100))
        assert torch.equal(gated[0, 400:], torch.zeros(600))
        assert torch.equal(gated[1], torch.zeros(1000))
        assert frac == [pytest.approx(0.7), pytest.approx(1.0)]

    def test_input_is_not_mutated(self):
        from egs3.conversational.tts.src.chunked_inference import silence_gate

        wav = torch.ones(1, 100)
        original = wav.clone()
        silence_gate(wav, FS, threshold=0.15, speech_regions_fn=lambda *a: [])
        assert torch.equal(wav, original)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("faster_whisper") is not None,
        reason="faster-whisper installed; the fallback error cannot trigger",
    )
    def test_default_vad_missing_dependency_is_a_clear_error(self):
        from egs3.conversational.tts.src.chunked_inference import silence_gate

        with pytest.raises(RuntimeError, match="faster-whisper"):
            silence_gate(torch.zeros(1, 100), FS, threshold=0.15)


class TestMatchActiveRms:
    def test_scales_channel_to_target(self):
        from egs3.conversational.tts.src.chunked_inference import match_active_rms

        wav = torch.full((1, FS), 0.2)
        out, gains = match_active_rms(wav, [0.4], FS)
        assert gains == [pytest.approx(2.0, abs=1e-6)]
        assert torch.allclose(out, torch.full((1, FS), 0.4), atol=1e-6)

    def test_silent_channel_and_none_target_pass_through(self):
        from egs3.conversational.tts.src.chunked_inference import match_active_rms

        wav = torch.stack([torch.zeros(FS), torch.full((FS,), 0.2)])
        out, gains = match_active_rms(wav, [0.5, None], FS)
        assert gains == [1.0, 1.0]
        assert torch.equal(out, wav)

    def test_gain_is_clamped(self):
        from egs3.conversational.tts.src.chunked_inference import match_active_rms

        wav = torch.full((1, FS), 0.01)
        out, gains = match_active_rms(wav, [0.5], FS)
        assert gains == [pytest.approx(10.0)]
        assert torch.allclose(out, torch.full((1, FS), 0.1), atol=1e-6)


class TestCondConfigValidation:
    def test_threshold_without_gate_is_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset,
            tmp_path / "infer",
            {"turns": 2, "cond_gate_threshold": 0.3},
        )
        with pytest.raises(
            ValueError, match="cond_gate_threshold requires cond_silence_gate"
        ):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_unknown_keys_still_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset, tmp_path / "infer", {"turns": 2, "cond_loudness_nrom": True}
        )
        with pytest.raises(ValueError, match="unknown chunk keys"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )


class TestConditioningHygiene:
    """The transform applies to what round k+1 SEES, never to what is
    WRITTEN - the mirror image of the cross-fade contract.  Pinned to the
    v1 zeros fill; the room-tone default is covered by
    TestConditioningHygieneV2."""

    CHUNK = {
        "turns": 2,
        "cond_silence_gate": True,
        "cond_gate_fill": "zeros",
        "cond_loudness_norm": True,
    }

    def _speech_first_half(self, ch_wav, fs, threshold):
        return [(0, ch_wav.shape[0] // 2)]

    def _run_spied(self, testset, tiny_model, tmp_path, monkeypatch, chunk):
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            out = real(model, vocoder, items, **kwargs)
            captured.append((items, out))
            return out

        monkeypatch.setattr(ci, "generate_batch", spy)
        cfg = _chunked_config(testset, tmp_path / "infer", chunk)
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
            speech_regions_fn=self._speech_first_half,
        )
        return tmp_path / "infer" / "valid", captured

    def test_round1_prompt_is_gated_and_normalized(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        test_dir, captured = self._run_spied(
            testset, tiny_model, tmp_path, monkeypatch, self.CHUNK
        )
        round1_items, _ = captured[-1]
        item = round1_items[0]
        prompt = item.speech[:, : item.prompt_frames * HOP].cpu()

        # Locate the raw round-0 output this prompt was derived from.
        raw = None
        for items, out in captured[:-1]:
            wavs = out[0][0]
            if wavs.shape[1] // HOP == item.prompt_frames:
                raw = wavs
        assert raw is not None

        # Longhand expected transform: zero the non-speech half, then scale
        # each channel to the REAL prompt's active-frame RMS.
        expected = raw.clone()
        expected[:, expected.shape[1] // 2 :] = 0.0
        for ch in range(expected.shape[0]):
            ref, sr = _read_wav(test_dir / f"prompt/000_ch{ch}.wav")
            assert sr == FS
            target = _expected_active_rms(torch.as_tensor(ref, dtype=torch.float32), FS)
            got = _expected_active_rms(expected[ch], FS)
            if target is None or got is None:
                continue
            gain = min(max(target / got, 0.1), 10.0)
            expected[ch] *= gain
        trimmed = expected[:, : item.prompt_frames * HOP]
        assert torch.allclose(prompt, trimmed, atol=1e-5)
        # The transform must actually bite in this setup: the second half is
        # zeroed, so prompt != raw.
        assert not torch.equal(prompt, raw[:, : item.prompt_frames * HOP])

    def test_written_wav_keeps_the_raw_chunk_audio(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        test_dir, captured = self._run_spied(
            testset, tiny_model, tmp_path, monkeypatch, self.CHUNK
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        chunk0_samples = meta["chunking"]["chunks"][0]["gen_frames"] * HOP
        raw0 = None
        for items, out in captured:
            wavs = out[0][0]
            if wavs.shape[1] == chunk0_samples:
                raw0 = wavs
        assert raw0 is not None
        for ch in range(raw0.shape[0]):
            data, _ = _read_wav(test_dir / f"wav/000_ch{ch}.wav")
            written = torch.as_tensor(data[:chunk0_samples], dtype=torch.float32)
            assert torch.allclose(written, raw0[ch], atol=2e-4)

    def test_meta_records_the_knobs_and_per_round_stats(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        test_dir, _ = self._run_spied(
            testset, tiny_model, tmp_path, monkeypatch, self.CHUNK
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        chunking = meta["chunking"]
        assert chunking["cond_silence_gate"] is True
        assert chunking["cond_gate_threshold"] == 0.15
        assert chunking["cond_loudness_norm"] is True
        chunks = chunking["chunks"]
        assert "conditioning" not in chunks[0]
        cond = chunks[1]["conditioning"]
        assert len(cond["gains"]) == meta["num_channels"]
        assert len(cond["gated_frac"]) == meta["num_channels"]
        assert all(0.0 <= f <= 1.0 for f in cond["gated_frac"])
        assert all(g > 0 for g in cond["gains"])

    def test_defaults_keep_the_knobs_off_in_meta(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(testset, tmp_path / "infer", {"turns": 2})
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        meta = json.loads(
            (tmp_path / "infer" / "valid" / "meta/000.json").read_text("utf-8")
        )
        chunking = meta["chunking"]
        assert chunking["cond_silence_gate"] is False
        assert chunking["cond_loudness_norm"] is False
        assert all("conditioning" not in c for c in chunking["chunks"])


# --------------------------------------------------------------------------- #
# Room tone: in-domain silence fill (v2 of the conditioning hygiene)
# --------------------------------------------------------------------------- #
class TestRoomTone:
    def test_snippet_is_the_quiet_frames_in_order(self):
        from egs3.conversational.tts.src.chunked_inference import room_tone

        frame = FS // 50
        loud = torch.full((10 * frame,), 0.5)
        quiet = torch.linspace(-5e-4, 5e-4, 6 * frame)
        tone = room_tone(torch.cat([loud, quiet]), FS)
        assert torch.equal(tone, quiet)

    def test_fallback_rescales_the_quietest_frame(self):
        from egs3.conversational.tts.src.chunked_inference import room_tone

        frame = FS // 50
        wav = torch.cat([torch.full((frame,), 0.5), torch.full((frame,), 0.01)])
        tone = room_tone(wav, FS)
        assert tone.shape[0] == frame
        got_rms = float(tone.pow(2).mean().sqrt())
        assert got_rms == pytest.approx(6e-4, rel=1e-4)

    def test_tile_to_repeats_and_trims(self):
        from egs3.conversational.tts.src.chunked_inference import tile_to

        snippet = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(
            tile_to(snippet, 7), torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
        )
        assert torch.equal(tile_to(torch.zeros(0), 4), torch.zeros(4))


class TestSilenceGateFill:
    def test_fill_replaces_non_speech_instead_of_zeros(self):
        from egs3.conversational.tts.src.chunked_inference import silence_gate

        wav = torch.randn(2, 1000)
        tone0 = torch.tensor([1e-4, -1e-4])
        tone1 = torch.tensor([2e-4])

        def regions(ch_wav, fs, threshold):
            return [(100, 400)]

        gated, frac = silence_gate(
            wav,
            FS,
            threshold=0.15,
            speech_regions_fn=regions,
            fill=[tone0, tone1],
        )
        assert torch.equal(gated[0, 100:400], wav[0, 100:400])
        expected0 = tone0.repeat(500)  # tiled across the full length
        assert torch.equal(gated[0, :100], expected0[:100])
        assert torch.equal(gated[0, 400:], expected0[400:])
        expected1 = tone1.repeat(1000)
        assert torch.equal(gated[1, 400:], expected1[400:])
        assert frac == [pytest.approx(0.7), pytest.approx(0.7)]

    def test_no_fill_keeps_v1_zeros(self):
        from egs3.conversational.tts.src.chunked_inference import silence_gate

        wav = torch.ones(1, 100)
        gated, _ = silence_gate(
            wav, FS, threshold=0.15, speech_regions_fn=lambda *a: []
        )
        assert torch.equal(gated, torch.zeros(1, 100))


class TestPromptBlocksFill:
    def test_default_stays_zeros(self):
        from egs3.conversational.tts.src.external_inference import _prompt_blocks

        wavs = [torch.full((100,), 0.3), torch.full((80,), 0.4)]
        blocks = _prompt_blocks(wavs, 2)
        assert torch.equal(blocks[0][1], torch.zeros(100))
        assert torch.equal(blocks[1][0], torch.zeros(80))

    def test_room_tone_fill_uses_each_channels_own_prompt(self):
        from egs3.conversational.tts.src.chunked_inference import room_tone, tile_to
        from egs3.conversational.tts.src.external_inference import _prompt_blocks

        frame = FS // 50
        # ch0 prompt: loud speech then a distinctive quiet tail.
        quiet0 = torch.full((frame,), 3e-4)
        wav0 = torch.cat([torch.full((3 * frame,), 0.5), quiet0])
        # ch1 prompt: all loud (exercises the fallback path).
        wav1 = torch.full((2 * frame,), 0.2)
        blocks = _prompt_blocks([wav0, wav1], 2, fill="room_tone", fs=FS)
        # Own rows are the raw prompts, untouched.
        assert torch.equal(blocks[0][0], wav0)
        assert torch.equal(blocks[1][1], wav1)
        # Off rows carry the OTHER channel's own room tone, tiled.
        assert torch.equal(blocks[0][1], tile_to(room_tone(wav1, FS), wav0.shape[0]))
        assert torch.equal(blocks[1][0], tile_to(room_tone(wav0, FS), wav1.shape[0]))

    def test_room_tone_fill_requires_fs(self):
        from egs3.conversational.tts.src.external_inference import _prompt_blocks

        with pytest.raises(ValueError, match="fs"):
            _prompt_blocks([torch.zeros(100)], 1, fill="room_tone")

    def test_unknown_fill_rejected(self):
        from egs3.conversational.tts.src.external_inference import _prompt_blocks

        with pytest.raises(ValueError, match="fill"):
            _prompt_blocks([torch.zeros(100)], 1, fill="pink_noise")


class TestV2ConfigValidation:
    def test_gate_fill_without_gate_is_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset,
            tmp_path / "infer",
            {"turns": 2, "cond_gate_fill": "zeros"},
        )
        with pytest.raises(
            ValueError, match="cond_gate_fill requires cond_silence_gate"
        ):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_bad_gate_fill_value_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(
            testset,
            tmp_path / "infer",
            {"turns": 2, "cond_silence_gate": True, "cond_gate_fill": "white"},
        )
        with pytest.raises(ValueError, match="cond_gate_fill must be"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_bad_prompt_fill_rejected(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(testset, tmp_path / "infer", {"turns": 2})
        cfg.prompt_fill = "white"
        with pytest.raises(ValueError, match="prompt_fill"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )


class TestConditioningHygieneV2:
    """Room tone in both zero-fill sites: the gate and the round-0 prompt."""

    CHUNK = {
        "turns": 2,
        "cond_silence_gate": True,
        "cond_loudness_norm": True,
    }

    def _speech_first_half(self, ch_wav, fs, threshold):
        return [(0, ch_wav.shape[0] // 2)]

    def _run_spied(self, testset, tiny_model, tmp_path, monkeypatch, chunk, **over):
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            out = real(model, vocoder, items, **kwargs)
            captured.append((items, out))
            return out

        monkeypatch.setattr(ci, "generate_batch", spy)
        cfg = _chunked_config(testset, tmp_path / "infer", chunk, **over)
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
            speech_regions_fn=self._speech_first_half,
        )
        return tmp_path / "infer" / "valid", captured

    def test_round0_prompt_off_rows_are_room_tone(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        from egs3.conversational.tts.src.chunked_inference import room_tone, tile_to

        test_dir, captured = self._run_spied(
            testset,
            tiny_model,
            tmp_path,
            monkeypatch,
            self.CHUNK,
            prompt_fill="room_tone",
        )
        # Round-0 item for dialogue 000: its prompt region is the two
        # concatenated blocks; the off rows must be non-zero room tone.
        item = captured[0][0][0]
        prompt = item.speech[:, : item.prompt_frames * HOP].cpu()
        ref0, sr = _read_wav(test_dir / "prompt/000_ch0.wav")
        assert sr == FS
        w0 = torch.as_tensor(ref0, dtype=torch.float32)
        # Block 0 spans [0, len(w0)); its row 1 is ch1's room tone.
        ref1, _ = _read_wav(test_dir / "prompt/000_ch1.wav")
        w1 = torch.as_tensor(ref1, dtype=torch.float32)
        expected_off = tile_to(room_tone(w1, FS), w0.shape[0])
        n0 = min(w0.shape[0], prompt.shape[1])
        assert torch.allclose(prompt[1, :n0], expected_off[:n0], atol=2e-4)
        assert not torch.equal(prompt[1, :n0], torch.zeros(n0))

    def test_round1_gate_fills_with_room_tone_not_zeros(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        from egs3.conversational.tts.src.chunked_inference import room_tone, tile_to

        test_dir, captured = self._run_spied(
            testset, tiny_model, tmp_path, monkeypatch, self.CHUNK
        )
        round1_items, _ = captured[-1]
        item = round1_items[0]
        prompt = item.speech[:, : item.prompt_frames * HOP].cpu()
        raw = None
        for items, out in captured[:-1]:
            wavs = out[0][0]
            if wavs.shape[1] // HOP == item.prompt_frames:
                raw = wavs
        assert raw is not None
        # Longhand: fill the non-speech half with the channel's prompt room
        # tone, then normalize to the prompt's active RMS.
        expected = raw.clone()
        half = expected.shape[1] // 2
        for ch in range(expected.shape[0]):
            ref, _ = _read_wav(test_dir / f"prompt/000_ch{ch}.wav")
            w = torch.as_tensor(ref, dtype=torch.float32)
            tone = tile_to(room_tone(w, FS), expected.shape[1])
            expected[ch, half:] = tone[half:]
            target = _expected_active_rms(w, FS)
            got = _expected_active_rms(expected[ch], FS)
            if target is None or got is None:
                continue
            gain = min(max(target / got, 0.1), 10.0)
            expected[ch] *= gain
        trimmed = expected[:, : item.prompt_frames * HOP]
        assert torch.allclose(prompt, trimmed, atol=1e-5)

    def test_meta_records_the_fill_choices(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        test_dir, _ = self._run_spied(
            testset,
            tiny_model,
            tmp_path,
            monkeypatch,
            self.CHUNK,
            prompt_fill="room_tone",
        )
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["chunking"]["cond_gate_fill"] == "room_tone"
        assert meta["prompt"]["fill"] == "room_tone"

    def test_default_prompt_fill_is_zeros(self, testset, tiny_model, tmp_path):
        cfg = _chunked_config(testset, tmp_path / "infer", {"turns": 2})
        run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        meta = json.loads(
            (tmp_path / "infer" / "valid" / "meta/000.json").read_text("utf-8")
        )
        assert meta["prompt"]["fill"] == "zeros"


# --------------------------------------------------------------------------- #
# Measure battery on chunked output
# --------------------------------------------------------------------------- #
class TestChunkedMeasure:
    def test_full_battery_runs_on_chunked_output(self, testset, tiny_model, tmp_path):
        inference_dir = tmp_path / "infer"
        run_chunked_inference(
            _chunked_config(testset, inference_dir, {"turns": 2}),
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        results = measure(_external_metrics_config(inference_dir))

        # The chunked meta contract is layout-identical to the external
        # path's, so the measure stage runs unchanged - same assertion as
        # TestExternalMeasure.
        for suffix, expected in (
            ("ConversationASRMetric", ASR_SUMMARY_KEYS),
            ("SpeakerSimilarityMetric", SPEAKER_SUMMARY_KEYS),
            ("QualityMetric", QUALITY_SUMMARY_KEYS),
            ("NoReferenceInteractionMetric", INTERACTION_SUMMARY_KEYS),
        ):
            matches = [k for k in results if k.endswith(suffix)]
            assert len(matches) == 1, f"expected one {suffix} entry, got {matches}"
            summary = results[matches[0]]["valid"]
            assert not expected - set(summary)
            assert all(isinstance(v, float) or v is None for v in summary.values())


# --------------------------------------------------------------------------- #
# System dispatch
# --------------------------------------------------------------------------- #
def test_system_dispatch_literal_matches_mode():
    # system.py compares a literal so an SSSD run never imports this module;
    # this test pins the two names together (same doctrine as the external
    # path's pin test).
    from egs3.conversational.tts.src.system import CHUNKED_MODE as SYSTEM_LITERAL

    assert SYSTEM_LITERAL == CHUNKED_MODE


# --------------------------------------------------------------------------- #
# special_tokens conditioning round loop
# --------------------------------------------------------------------------- #
def _sptok_chunk(**over):
    chunk = {"turns": 2, "cond_format": "special_tokens", "cond_prev_sec": 1.0}
    chunk.update(over)
    return chunk


class TestSpecialTokensInfer:
    def _run(self, testset, tiny_model, inference_dir, chunk, **overrides):
        cfg = _chunked_config(testset, inference_dir, chunk, **overrides)
        stats = run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        return inference_dir / "valid", stats, cfg

    def test_meta_records_mode_and_conditioning_frames(
        self, testset, tiny_model, tmp_path
    ):
        out, stats, _ = self._run(testset, tiny_model, tmp_path / "o", _sptok_chunk())
        metas = [json.loads(p.read_text()) for p in sorted((out / "meta").glob("*"))]
        assert metas
        for meta in metas:
            ck = meta["chunking"]
            assert ck["cond_format"] == "special_tokens"
            assert ck["cond_prompt_sec"] == 8.0 and ck["cond_prev_sec"] == 1.0
            chunks = ck["chunks"]
            assert chunks[0]["prev_frames"] == 0
            # Conditioning prompt: shortest reference prompt (min-truncated),
            # capped at 8 s, floored to whole hops - recomputed longhand from
            # the prompt files themselves.
            record_prompts = meta["prompt"]["turns"]
            shortest = min(p["duration_sec"] for p in record_prompts)
            expect_p = int(min(shortest, 8.0) * FS) // HOP
            assert all(c["prompt_frames"] == expect_p for c in chunks)
            assert meta["prompt"]["total_frames"] == expect_p
            # Rounds after the first: 1.0 s tail = 93 frames (93.75 floored),
            # unless less audio was generated (not the case at turns: 2).
            for c in chunks[1:]:
                assert c["prev_frames"] == int(1.0 * FS) // HOP == 93

    def test_prompt_below_trained_floor_flag_and_warning(
        self, tmp_path, caplog
    ):
        # A dedicated testset with one dialogue whose shortest reference
        # prompt is below the 3 s trained own-speech floor ("shortp", min
        # 2.0 s) and one at/above it ("longp", min 3.5 s) - the default
        # `testset` fixture's prompts are all < 3 s, so it cannot exercise
        # the "absent when P >= floor" side of this contract.
        dialogues = {
            "shortp": (
                ["abc def", "bead cab", "chad face"],
                [
                    ("test-clean/1/1/a.flac", "abc", 2.0),
                    ("test-clean/2/2/b.flac", "de", 2.5),
                ],
            ),
            "longp": (
                ["gaff bead", "haji dead"],
                [
                    ("test-clean/3/3/c.flac", "chad", 3.5),
                    ("test-clean/4/4/d.flac", "fig", 4.0),
                ],
            ),
        }
        ts = _write_testset(tmp_path, dialogues=dialogues)
        model = build_tiny(ts["vocab"])
        cfg = _chunked_config(ts, tmp_path / "o", _sptok_chunk())
        with caplog.at_level("WARNING"):
            run_chunked_inference(
                cfg,
                training_config=ts["training_config"],
                model=model,
                vocoder=FakeVocoder(),
            )
        metas = {
            p.stem: json.loads(p.read_text())
            for p in sorted((tmp_path / "o" / "valid" / "meta").glob("*"))
        }
        assert set(metas) == {"shortp", "longp"}
        assert metas["shortp"]["chunking"]["prompt_below_trained_floor"] is True
        assert "prompt_below_trained_floor" not in metas["longp"]["chunking"]

        warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "shortp" in r.getMessage()
        ]
        assert warnings, caplog.text
        assert f"{SPECIAL_TOKENS_PROMPT_FLOOR_SEC:.1f}s" in warnings[0].getMessage()
        assert not any("longp" in r.getMessage() for r in caplog.records)

    def test_prompt_files_on_disk_are_full_length(
        self, testset, tiny_model, tmp_path
    ):
        out, _, _ = self._run(testset, tiny_model, tmp_path / "o", _sptok_chunk())
        metas = [json.loads(p.read_text()) for p in sorted((out / "meta").glob("*"))]
        for meta in metas:
            for p, entry in zip(meta["prompt"]["turns"], meta["channels"]):
                wav, _ = _read_wav(out / entry["prompt_wav"])
                assert abs(wav.shape[0] / FS - p["duration_sec"]) < 0.05

    def test_zero_prev_sec_is_reanchor_no_history_frames(
        self, testset, tiny_model, tmp_path
    ):
        out, _, _ = self._run(
            testset, tiny_model, tmp_path / "o", _sptok_chunk(cond_prev_sec=0.0)
        )
        metas = [json.loads(p.read_text()) for p in sorted((out / "meta").glob("*"))]
        for meta in metas:
            assert all(
                c["prev_frames"] == 0 for c in meta["chunking"]["chunks"]
            )

    def test_wav_is_concat_of_chunk_regions(self, testset, tiny_model, tmp_path):
        out, _, _ = self._run(testset, tiny_model, tmp_path / "o", _sptok_chunk())
        metas = [json.loads(p.read_text()) for p in sorted((out / "meta").glob("*"))]
        for meta in metas:
            total = sum(c["gen_frames"] for c in meta["chunking"]["chunks"]) * HOP
            wav, _ = _read_wav(out / meta["channels"][0]["gen_wav"])
            assert wav.shape[0] == total

    def test_special_mode_differs_from_transcripts_mode(
        self, testset, tiny_model, tmp_path
    ):
        out_a, _, _ = self._run(testset, tiny_model, tmp_path / "a", {"turns": 2})
        out_b, _, _ = self._run(testset, tiny_model, tmp_path / "b", _sptok_chunk())
        name = sorted((out_a / "wav").glob("*"))[0].name
        wa, _ = _read_wav(out_a / "wav" / name)
        wb, _ = _read_wav(out_b / "wav" / name)
        assert wa.shape != wb.shape or not (wa == wb).all()

    def test_round1_conditioning_is_p_plus_tail_of_h(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        # The meta-only assertions above pin frame COUNTS; this pins the
        # actual GenerationItem.speech content: P is byte-identical between
        # rounds (fixed window, never re-derived), and H is the TAIL - not
        # the head - of round 0's generated audio.
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            out = real(model, vocoder, items, **kwargs)
            captured.append((items, out))
            return out

        monkeypatch.setattr(ci, "generate_batch", spy)
        self._run(testset, tiny_model, tmp_path / "infer", _sptok_chunk())

        # No batching block -> `plan_batches` returns `[[i] for i in
        # indices]` verbatim (its documented no-batching path), so call
        # order is exactly the (shard-local) dialogue order: "000" (3
        # turns, 2 chunks) then "001" (2 turns, 1 chunk) at round 0, then
        # "000" again at round 1 - "001" never gets a second chunk at
        # turns=2.  (The fabricated prompts are all the same fixed-frequency
        # sine per `_write_flac`, so content alone cannot disambiguate
        # dialogues here - order is the only reliable handle.)
        assert len(captured) == 3
        item0, wav0 = captured[0][0][0], captured[0][1][0][0]
        item1 = captured[2][0][0]

        p_total = item1.prompt_frames  # p_frames + prev_frames, the TOTAL
        p_frames = item0.prompt_frames  # round 0's P is prev_frames=0, so
        # its own prompt_frames IS p_frames.
        prev_frames = p_total - p_frames
        assert prev_frames > 0

        # P is byte-identical between round 0 and round 1 (fixed window,
        # never re-derived from generated audio).
        assert torch.equal(
            item0.speech[:, : p_frames * HOP].cpu(),
            item1.speech[:, : p_frames * HOP].cpu(),
        )
        # H is the TAIL of round 0's generated audio (not the head).
        tail = wav0[:, wav0.shape[1] - prev_frames * HOP :].cpu()
        h_span = item1.speech[:, p_frames * HOP : p_total * HOP].cpu()
        assert torch.equal(h_span, tail)

        # The generated region past the conditioning span is still zeros.
        gen_region = item1.speech[:, p_total * HOP :]
        assert torch.equal(gen_region, torch.zeros_like(gen_region))

    def test_hygiene_engages_on_h_only_not_round0(
        self, testset, tiny_model, tmp_path
    ):
        # H (round >= 1's generated-audio tail) goes through the hygiene
        # knobs; P (round 0's parallel real-prompt span) is exempt exactly
        # like the transcripts format's prompt segment - this exercises the
        # special-mode anchor construction from full prompt wavs, which the
        # meta-only frame-count tests above never touch.
        out, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "o",
            _sptok_chunk(cond_loudness_norm=True),
        )
        metas = [json.loads(p.read_text()) for p in sorted((out / "meta").glob("*"))]
        assert metas
        n_round_ge1 = 0
        for meta in metas:
            n = meta["num_channels"]
            for c in meta["chunking"]["chunks"]:
                if c["round"] == 0:
                    assert "conditioning" not in c
                else:
                    assert len(c["conditioning"]["gains"]) == n
                    n_round_ge1 += 1
        assert n_round_ge1 > 0

    def test_round1_text_prefix_is_speaker_prompt_then_prev_chunk(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        # Pins that the preprocessor receives P-only prompt_frames (not the
        # P+H total): the first p_frames entries of every branch's text row
        # must be <speaker_prompt>, the next prev_frames must be
        # <prev_chunk> - same spy/capture pattern as
        # test_round1_conditioning_is_p_plus_tail_of_h above.
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            out = real(model, vocoder, items, **kwargs)
            captured.append((items, out))
            return out

        monkeypatch.setattr(ci, "generate_batch", spy)
        self._run(testset, tiny_model, tmp_path / "infer", _sptok_chunk())

        assert len(captured) == 3
        item0 = captured[0][0][0]
        item1 = captured[2][0][0]

        p_total = item1.prompt_frames  # p_frames + prev_frames, the TOTAL
        p_frames = item0.prompt_frames  # round 0's P is prev_frames=0, so
        # its own prompt_frames IS p_frames.
        prev_frames = p_total - p_frames
        assert prev_frames > 0

        token2id = make_token2id(
            Path(testset["vocab"]).read_text(encoding="utf-8").splitlines()
        )
        sp_id = token2id[SPEAKER_PROMPT_TOKEN]
        pc_id = token2id[PREV_CHUNK_TOKEN]

        text = item1.text.cpu()
        for branch in range(text.shape[0]):
            row = text[branch]
            assert torch.equal(
                row[:p_frames], torch.full((p_frames,), sp_id, dtype=row.dtype)
            )
            assert torch.equal(
                row[p_frames : p_frames + prev_frames],
                torch.full((prev_frames,), pc_id, dtype=row.dtype),
            )

    def test_special_tokens_rejects_legacy_vocab(
        self, testset, tiny_model, tmp_path
    ):
        # A 2-token vocab file next to the testset's real one.
        legacy = tmp_path / "legacy_vocab.txt"
        tokens = Path(testset["vocab"]).read_text().splitlines()
        assert tokens[-4:] == list(NEW_TOKENS)
        legacy.write_text("\n".join(tokens[:-4] + tokens[-4:-2]) + "\n")
        cfg = _chunked_config(testset, tmp_path / "o", _sptok_chunk())
        tc = OmegaConf.create(
            OmegaConf.to_container(testset["training_config"], resolve=True)
        )
        tc.dataset.preprocessor.token_list = str(legacy)
        with pytest.raises(ValueError, match="special_tokens needs a vocab"):
            run_chunked_inference(
                cfg, training_config=tc, model=tiny_model, vocoder=FakeVocoder()
            )

    def test_special_tokens_rejects_room_tone_prompt_fill(
        self, testset, tiny_model, tmp_path
    ):
        cfg = _chunked_config(
            testset, tmp_path / "o", _sptok_chunk(), prompt_fill="room_tone"
        )
        with pytest.raises(ValueError, match="prompt_fill has no effect"):
            run_chunked_inference(
                cfg,
                training_config=testset["training_config"],
                model=tiny_model,
                vocoder=FakeVocoder(),
            )

    def test_explicit_transcripts_mode_is_bit_identical_to_default(
        self, testset, tiny_model, tmp_path
    ):
        out_a, _, _ = self._run(testset, tiny_model, tmp_path / "a", {"turns": 2})
        out_b, _, _ = self._run(
            testset, tiny_model, tmp_path / "b",
            {"turns": 2, "cond_format": "transcripts"},
        )
        names = sorted(p.name for p in (out_a / "wav").glob("*"))
        assert names
        for name in names:
            # Byte-identical, not merely close: an explicit
            # `cond_format: transcripts` must take the exact same code path
            # as omitting the key (rtol=0.0, atol=0.0 - house precedent in
            # test_preprocessing_parity.py).
            torch.testing.assert_close(
                _read_wav(out_a / "wav" / name),
                _read_wav(out_b / "wav" / name),
                rtol=0.0,
                atol=0.0,
            )


# --------------------------------------------------------------------------- #
# text_format: timestamps (Mode T) round loop
# --------------------------------------------------------------------------- #
def _mode_t_chunk(**over):
    chunk = {
        "turns": 2,
        "cond_format": "special_tokens",
        "cond_prev_sec": 1.0,
        "text_format": "timestamps",
    }
    chunk.update(over)
    return chunk


class TestTimestampInfer:
    @pytest.fixture
    def testset(self, tmp_path):
        # Mode T writes <turn_fill>, the fifth (timestamp-era) new token; the
        # shared fixture vocab stops at the four special-token-era ones, so
        # this class needs the extended vocab (and a model sized for it).
        ts = _write_testset(tmp_path)
        vocab = Path(ts["vocab"])
        tokens = vocab.read_text("utf-8").splitlines()
        assert tokens[-4:] == list(NEW_TOKENS)
        vocab.write_text(
            "\n".join(tokens + [TURN_FILL_TOKEN]) + "\n", encoding="utf-8"
        )
        return ts

    @pytest.fixture
    def tiny_model(self, testset):
        return build_tiny(testset["vocab"])

    def _run(self, testset, tiny_model, inference_dir, chunk, **overrides):
        cfg = _chunked_config(testset, inference_dir, chunk, **overrides)
        stats = run_chunked_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        return inference_dir / "valid", stats, cfg

    def _metas(self, out):
        metas = [
            json.loads(p.read_text("utf-8")) for p in sorted((out / "meta").glob("*"))
        ]
        assert metas
        return {m["window_id"]: m for m in metas}

    def test_meta_records_format_and_layout(self, testset, tiny_model, tmp_path):
        out, _, _ = self._run(testset, tiny_model, tmp_path / "o", _mode_t_chunk())
        for wid, meta in self._metas(out).items():
            ck = meta["chunking"]
            assert ck["text_format"] == "timestamps" and ck["turn_gap_sec"] == 0.4
            assert meta["turn_times"] == "layout"
            assert meta["layout"]["gap_sec"] == 0.4
            lay = meta["layout"]["turns"]
            # one layout entry per dialogue turn (DIALOGUES: "000" 3, "001" 2)
            assert len(lay) == {"000": 3, "001": 2}[wid]
            assert [t["channel"] for t in lay] == [
                t["channel"] for t in meta["turns"]
            ]
            # sequential, non-overlapping, gap-separated (the realized gap is
            # the requested one rounded to the frame grid)
            for prev, cur in zip(lay, lay[1:]):
                assert cur["start"] >= prev["end"]
                assert abs((cur["start"] - prev["end"]) - 0.4) < 2 / FRAMES_PER_SECOND
            for c in ck["chunks"]:
                assert c["target_frames"] == c["gen_frames"]

    def test_target_spans_tile_the_layout(self, testset, tiny_model, tmp_path):
        # Chunk k's target starts exactly where chunk k-1's ended: the round
        # loop's t0/frames must partition the synthesized timeline, or the
        # written wav would not be the layout it claims to realize.
        out, _, _ = self._run(testset, tiny_model, tmp_path / "o", _mode_t_chunk())
        for meta in self._metas(out).values():
            cursor = 0
            for c in meta["chunking"]["chunks"]:
                assert c["target_t0_sec"] == pytest.approx(
                    cursor / FRAMES_PER_SECOND, abs=1e-6
                )
                cursor += c["target_frames"]
            # Every layout turn lands inside the span the chunks cover.
            assert meta["layout"]["turns"][-1]["end"] <= cursor / FRAMES_PER_SECOND

    def test_chunk_plans_identical_to_mode_o(self, testset, tiny_model, tmp_path):
        out_o, _, _ = self._run(testset, tiny_model, tmp_path / "o", _sptok_chunk())
        out_t, _, _ = self._run(testset, tiny_model, tmp_path / "t", _mode_t_chunk())
        mo, mt = self._metas(out_o), self._metas(out_t)
        assert mo.keys() == mt.keys()
        for wid in mo:
            ranges_o = [
                (c["turn_start"], c["turn_end"]) for c in mo[wid]["chunking"]["chunks"]
            ]
            ranges_t = [
                (c["turn_start"], c["turn_end"]) for c in mt[wid]["chunking"]["chunks"]
            ]
            assert ranges_o == ranges_t
            # predicted seconds (the duration policy) stay gap-free
            assert mo[wid]["duration"] == mt[wid]["duration"]
            assert [c["predicted_sec"] for c in mo[wid]["chunking"]["chunks"]] == [
                c["predicted_sec"] for c in mt[wid]["chunking"]["chunks"]
            ]

    def test_generated_audio_matches_layout_frames(self, testset, tiny_model, tmp_path):
        out, _, _ = self._run(testset, tiny_model, tmp_path / "o", _mode_t_chunk())
        for meta in self._metas(out).values():
            total = sum(c["target_frames"] for c in meta["chunking"]["chunks"]) * HOP
            wav, _ = _read_wav(out / meta["channels"][0]["gen_wav"])
            assert wav.shape[0] == total
            # gaps lengthen each chunk relative to the gap-free estimate
            for c in meta["chunking"]["chunks"]:
                assert c["target_frames"] > round(c["predicted_sec"] * FS / HOP) - 2

    def test_text_stream_is_one_token_per_frame(
        self, testset, tiny_model, tmp_path, monkeypatch
    ):
        import egs3.conversational.tts.src.chunked_inference as ci

        captured = []
        real = ci.generate_batch

        def spy(model, vocoder, items, **kwargs):
            captured.extend(items)
            return real(model, vocoder, items, **kwargs)

        monkeypatch.setattr(ci, "generate_batch", spy)
        self._run(testset, tiny_model, tmp_path / "o", _mode_t_chunk())
        assert captured
        for item in captured:
            # Mode T text covers P + H + target frame for frame, so every
            # branch has the same length and nothing is pad (-1).
            assert item.text.shape[1] == item.total_frames
            assert (item.text[:, item.prompt_frames :] >= 0).all()

    def test_zero_gap_layout(self, testset, tiny_model, tmp_path):
        out, _, _ = self._run(
            testset, tiny_model, tmp_path / "o", _mode_t_chunk(turn_gap_sec=0.0)
        )
        for meta in self._metas(out).values():
            assert meta["chunking"]["turn_gap_sec"] == 0.0
            lay = meta["layout"]["turns"]
            for prev, cur in zip(lay, lay[1:]):
                assert abs(cur["start"] - prev["end"]) < 1e-6

    def test_mode_t_differs_from_mode_o(self, testset, tiny_model, tmp_path):
        out_o, _, _ = self._run(testset, tiny_model, tmp_path / "o", _sptok_chunk())
        out_t, _, _ = self._run(testset, tiny_model, tmp_path / "t", _mode_t_chunk())
        name = sorted((out_o / "wav").glob("*"))[0].name
        wo, _ = _read_wav(out_o / "wav" / name)
        wt, _ = _read_wav(out_t / "wav" / name)
        assert wo.shape != wt.shape or not (wo == wt).all()

    def test_warns_when_gaps_push_a_chunk_past_target_sec(
        self, testset, tiny_model, tmp_path, caplog
    ):
        # The packed ceiling the operator sets from the trained window is
        # gap-free, but Mode T generates the packed span PLUS one gap per
        # turn.  `oversized` stays the gap-free test on purpose (its rows are
        # read beside the Mode O ones), so the overrun has to surface as a
        # warning.  Same packing in both runs - only turn_gap_sec differs.
        chunk = {
            "cond_format": "special_tokens",
            "cond_prev_sec": 1.0,
            "text_format": "timestamps",
            "target_sec": 11.0,
        }
        with caplog.at_level("WARNING"):
            out, _, _ = self._run(
                testset, tiny_model, tmp_path / "gap", dict(chunk, turn_gap_sec=0.4)
            )
        warned = [
            r.getMessage()
            for r in caplog.records
            if r.levelname == "WARNING" and "target_sec" in r.getMessage()
        ]
        assert len(warned) == 1, caplog.text
        assert warned[0].startswith("000: chunk 0 realizes ")
        # ...and it describes a REAL overrun: gap-free within budget,
        # realized past it, derived from the meta rather than from the same
        # arithmetic the warning itself used.
        meta = self._metas(out)["000"]
        entry = meta["chunking"]["chunks"][0]
        assert entry["predicted_sec"] <= 11.0
        assert entry["target_frames"] / FRAMES_PER_SECOND > 11.0
        # The flag itself must NOT move - it stays the cross-mode-comparable
        # gap-free test.
        assert meta["chunking"]["oversized"] == [False, False]

        caplog.clear()
        with caplog.at_level("WARNING"):
            self._run(
                testset, tiny_model, tmp_path / "nogap", dict(chunk, turn_gap_sec=0.0)
            )
        assert not [
            r for r in caplog.records if "target_sec" in r.getMessage()
        ], caplog.text

    def test_rejects_vocab_without_turn_fill(self, testset, tiny_model, tmp_path):
        # Mode T pads every turn's frame span with <turn_fill>; a vocab
        # without it must fail at the load gate, not with a raw KeyError
        # from the preprocessor deep inside the round loop.
        legacy = tmp_path / "no_turn_fill_vocab.txt"
        tokens = Path(testset["vocab"]).read_text("utf-8").splitlines()
        assert tokens[-1] == TURN_FILL_TOKEN
        legacy.write_text("\n".join(tokens[:-1]) + "\n", encoding="utf-8")
        tc = OmegaConf.create(
            OmegaConf.to_container(testset["training_config"], resolve=True)
        )
        tc.dataset.preprocessor.token_list = str(legacy)
        cfg = _chunked_config(testset, tmp_path / "o", _mode_t_chunk())
        with pytest.raises(ValueError, match=TURN_FILL_TOKEN):
            run_chunked_inference(
                cfg, training_config=tc, model=tiny_model, vocoder=FakeVocoder()
            )

    def test_rejects_frame_rate_mismatch(self, testset, tiny_model, tmp_path):
        # The Mode T text grid is hardwired to FRAMES_PER_SECOND by the
        # preprocessor, so an fs/hop that disagrees would silently desync the
        # text stream from the audio it describes.
        tc = OmegaConf.create(
            OmegaConf.to_container(testset["training_config"], resolve=True)
        )
        tc.hop_length = HOP * 2
        cfg = _chunked_config(testset, tmp_path / "o", _mode_t_chunk())
        with pytest.raises(ValueError, match="frame rate"):
            run_chunked_inference(
                cfg, training_config=tc, model=tiny_model, vocoder=FakeVocoder()
            )
