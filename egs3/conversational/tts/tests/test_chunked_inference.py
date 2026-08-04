"""Tests for the chunked external infer stage (src/chunked_inference.py).

Reuses the external-path fixtures: the fabricated CoVoMix2 tree, the tiny
random-init DiT, and FakeVocoder.  Duration expectations are written out
longhand (explicit per-speaker rates) so a shared bug in the formula under
test cannot pass both sides - same doctrine as test_external_testset.py.
"""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from egs3.conversational.tts.dataset.preprocessing.text import (
    OTHER_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
)
from egs3.conversational.tts.src.chunked_inference import (
    MODE as CHUNKED_MODE,
    call_turns,
    crossfade_concat,
    estimate_turn_secs,
    run_chunked_inference,
    split_turns,
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
