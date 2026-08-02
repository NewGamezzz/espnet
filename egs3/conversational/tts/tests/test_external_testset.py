"""Tests for the CoVoMix2 external-test-set adapter and its infer stage.

Fixture-based and CPU-only: a fabricated test-set tree (index JSON +
transcript files) plus fabricated mono "LibriSpeech" FLACs, the tiny
random-init DiT from the trainer suite, and the same FakeVocoder the SSSD
infer suite uses.

Text in the fixtures is restricted to the conftest vocab's charset (space
plus ``a``-``j``), because the adapter normalizes against the extended vocab
and the preprocessor fails loudly on OOV - the same contract the SSSD build
enforces at build time.

Duration expectations are computed with arithmetic INDEPENDENT of the
formula under test (explicit per-speaker rates written out longhand), so a
shared bug in that formula cannot pass both sides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    build_branch_texts,
    normalize_text,
    vocab_charset,
)
from egs3.conversational.tts.src.external_inference import (
    MODE,
    _prompt_blocks,
    run_external_inference,
)
from egs3.conversational.tts.src.external_testset import (
    DEFAULT_DURATION_SCALE,
    assign_shard,
    ExternalPrompt,
    ExternalRecord,
    _read_turns,
    estimate_duration_sec,
    load_covomix2_testset,
    plan_batches,
    select_records,
)
from egs3.conversational.tts.src.metrics.interaction_no_reference import (
    NoReferenceInteractionMetric,
)
from espnet3.systems.base.metric import measure

from .conftest import EXT_TOKENS
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)
from .test_e2e_eval import (
    ASR_SUMMARY_KEYS,
    INTERACTION_SUMMARY_KEYS,
    QUALITY_SUMMARY_KEYS,
    SPEAKER_SUMMARY_KEYS,
    _fake,
)
from .test_inference import FS, HOP, FakeVocoder, _read_wav, _write_flac

PROMPT_SR = 16000  # LibriSpeech's rate; resampled to FS by the loader


# --------------------------------------------------------------------------- #
# Transcript parsing (pure function)
# --------------------------------------------------------------------------- #
class TestReadTurns:
    def test_strict_alternation_from_speaker_one(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text("abc\ndef\nghi\njab\n", encoding="utf-8")
        turns = _read_turns(path, 2)
        assert [t.channel for t in turns] == [0, 1, 0, 1]
        assert [t.text for t in turns] == ["abc", "def", "ghi", "jab"]

    def test_blank_lines_dropped_before_alternation(self, tmp_path):
        # A stray blank line must NOT shift subsequent turns to the wrong
        # speaker - that would silently swap every reference text.
        path = tmp_path / "t.txt"
        path.write_text("abc\n\n  \ndef\nghi\n", encoding="utf-8")
        turns = _read_turns(path, 2)
        assert [t.channel for t in turns] == [0, 1, 0]
        assert [t.text for t in turns] == ["abc", "def", "ghi"]

    def test_turn_times_are_ordinals(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text("abc\ndef\n", encoding="utf-8")
        turns = _read_turns(path, 2)
        assert [t.start for t in turns] == [0.0, 1.0]
        # Stable-sorting the meta turns by "start" must recover conversation
        # order - the only thing the mixed-WER reference uses them for.
        assert sorted(turns, key=lambda t: t.start) == turns

    def test_empty_transcript_raises(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="no non-empty transcript lines"):
            _read_turns(path, 2)


# --------------------------------------------------------------------------- #
# Duration estimation (pure function)
# --------------------------------------------------------------------------- #
def _record(turn_texts, prompt_texts):
    turns = [
        Turn(i % 2, f"spk{i % 2}", text, float(i), float(i))
        for i, text in enumerate(turn_texts)
    ]
    prompts = [
        ExternalPrompt(channel=ch, audio_path=Path(f"/nonexistent/{ch}.flac"), text=t)
        for ch, t in enumerate(prompt_texts)
    ]
    return ExternalRecord(
        dialogue_id="d0", num_channels=2, turns=turns, prompts=prompts
    )


class TestDurationEstimate:
    def test_per_speaker_rates_applied_to_own_share(self):
        # ch0 turn "abcdefg" (7 chars), ch1 turn "abcdefgh" (8 chars).
        record = _record(["abcdefg", "abcdefgh"], ["abc", "de"])
        # Longhand, independent of the code: ch0 prompt 1.0s / 3 chars =
        # 1/3 s per char -> 7 * 1/3 = 2.333...; ch1 prompt 2.0s / 2 chars =
        # 1.0 s per char -> 8 * 1.0 = 8.0.  Total 10.333...
        got = estimate_duration_sec(record, [1.0, 2.0], duration_scale=1.0)
        assert got == pytest.approx(7 / 3 + 8.0)

    def test_scale_and_speed_are_multiplicative_and_inverse(self):
        record = _record(["abcdefg", "abcdefgh"], ["abc", "de"])
        base = estimate_duration_sec(record, [1.0, 2.0], duration_scale=1.0)
        scaled = estimate_duration_sec(
            record, [1.0, 2.0], duration_scale=2.0, speed=4.0
        )
        # F5's sense: larger speed is faster, therefore shorter.
        assert scaled == pytest.approx(base * 2.0 / 4.0)

    def test_default_scale_matches_documented_derivation(self):
        # (0.0735 / 0.0690) / 0.954, the two measured articulation rates and
        # the measured SSSD speech density.
        assert DEFAULT_DURATION_SCALE == pytest.approx(
            (0.0735 / 0.0690) / 0.954, abs=1e-3
        )

    def test_wrong_prompt_count_raises(self):
        record = _record(["abcdefg", "abcdefgh"], ["abc", "de"])
        with pytest.raises(ValueError, match="prompt durations"):
            estimate_duration_sec(record, [1.0])

    def test_degenerate_prompt_raises(self):
        record = _record(["abcdefg", "abcdefgh"], ["abc", "de"])
        with pytest.raises(ValueError, match="degenerate"):
            estimate_duration_sec(record, [0.0, 2.0])


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
class TestSelection:
    def _records(self, n):
        return [_record(["abc", "def"], ["abc", "de"]) for _ in range(n)]

    def test_band_excludes_out_of_regime_dialogues(self):
        durations = [10.0, 45.0, 61.0, 120.0]
        got, counts = select_records(
            self._records(4), durations, OmegaConf.create({"max_duration": 60.0})
        )
        assert got == [0, 1]
        assert counts == {"n_out_of_band": 2, "n_not_sampled": 0}

    def test_min_and_max_both_applied(self):
        durations = [5.0, 20.0, 90.0]
        got, counts = select_records(
            self._records(3),
            durations,
            OmegaConf.create({"min_duration": 10.0, "max_duration": 60.0}),
        )
        assert got == [1]
        assert counts["n_out_of_band"] == 2

    def test_subsample_is_seeded_sorted_and_reproducible(self):
        durations = [float(i) for i in range(20)]
        cfg = OmegaConf.create({"num_dialogues": 5, "seed": 0})
        first, counts = select_records(self._records(20), durations, cfg)
        second, _ = select_records(self._records(20), durations, cfg)
        assert first == second
        assert len(first) == 5
        assert first == sorted(first)
        # Not-sampled is NOT a failure count.
        assert counts == {"n_out_of_band": 0, "n_not_sampled": 15}

    def test_band_and_subsample_counted_separately(self):
        # The two exclusion reasons mean opposite things; a single "skipped"
        # number would report 18 failures here when only 3 are out of regime.
        durations = [float(i) for i in range(20)]  # 17, 18, 19 are out of band
        got, counts = select_records(
            self._records(20),
            durations,
            OmegaConf.create({"max_duration": 16.5, "num_dialogues": 4, "seed": 0}),
        )
        assert len(got) == 4
        assert counts == {"n_out_of_band": 3, "n_not_sampled": 13}

    def test_no_limits_keeps_everything(self):
        durations = [1.0, 2.0, 3.0]
        got, counts = select_records(self._records(3), durations, OmegaConf.create({}))
        assert got == [0, 1, 2]
        assert counts == {"n_out_of_band": 0, "n_not_sampled": 0}


# --------------------------------------------------------------------------- #
# Test-set loading
# --------------------------------------------------------------------------- #
DIALOGUES = {
    # key -> (transcript lines, [(prompt rel path, prompt text, seconds)])
    "000": (
        ["abc def", "bead cab", "chad face"],
        [("test-clean/1/1/a.flac", "abc", 1.0), ("test-clean/2/2/b.flac", "de", 2.0)],
    ),
    "001": (
        ["gaff bead", "haji dead"],
        [("test-clean/3/3/c.flac", "chad", 1.5), ("test-clean/4/4/d.flac", "fig", 1.0)],
    ),
}


def _write_testset(tmp_path) -> dict:
    testset_root = tmp_path / "testset"
    libri_root = tmp_path / "librispeech"
    (testset_root / "transcriptions").mkdir(parents=True)

    entries = []
    for key, (lines, prompts) in DIALOGUES.items():
        (testset_root / "transcriptions" / f"{key}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        entry = {"key": key, "text": f"transcriptions/{key}.txt"}
        for i, (rel, text, seconds) in enumerate(prompts, start=1):
            _write_flac(libri_root / rel, 1, seconds, PROMPT_SR)
            entry[f"audio_prompt_spk{i}"] = rel
            entry[f"audio_prompt_spk{i}_transcription"] = text
        entries.append(entry)

    (testset_root / "dailydialog-dialogue.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    return {
        "tmp_path": tmp_path,
        "testset_root": testset_root,
        "librispeech_root": libri_root,
        "vocab": vocab,
        "training_config": OmegaConf.create(
            {
                "recipe_dir": str(tmp_path),
                "sample_rate": FS,
                "hop_length": HOP,
                "dataset": {"preprocessor": {"token_list": str(vocab)}},
            }
        ),
    }


@pytest.fixture
def testset(tmp_path):
    return _write_testset(tmp_path)


class TestLoad:
    def _load(self, testset):
        return load_covomix2_testset(
            testset["testset_root"],
            testset["librispeech_root"],
            testset["vocab"],
        )

    def test_all_dialogues_loaded_with_alternating_channels(self, testset):
        records = self._load(testset)
        assert [r.dialogue_id for r in records] == ["000", "001"]
        assert [t.channel for t in records[0].turns] == [0, 1, 0]
        assert [p.channel for p in records[0].prompts] == [0, 1]

    def test_text_is_normalized_against_the_extended_vocab(self, testset):
        charset = vocab_charset(EXT_TOKENS)
        records = self._load(testset)
        expected = [normalize_text(line, charset) for line in DIALOGUES["000"][0]]
        assert [t.text for t in records[0].turns] == expected
        # Prompt transcriptions go through the SAME normalization.
        assert records[0].prompts[0].text == normalize_text("abc", charset)

    def test_channel_chars_counts_utf8_bytes_per_channel(self, testset):
        records = self._load(testset)
        record = records[0]
        expected = [0, 0]
        for turn in record.turns:
            expected[turn.channel] += len(turn.text.encode("utf-8"))
        assert record.channel_chars == expected

    def test_missing_prompt_audio_raises(self, testset):
        (testset["librispeech_root"] / "test-clean/1/1/a.flac").unlink()
        with pytest.raises(FileNotFoundError, match="prompt audio"):
            self._load(testset)


# --------------------------------------------------------------------------- #
# Prompt block assembly (pure function)
# --------------------------------------------------------------------------- #
class TestPromptBlocks:
    def test_own_row_carries_speech_others_are_silent(self):
        wavs = [torch.full((4,), 0.5), torch.full((6,), -0.25)]
        blocks = _prompt_blocks(wavs, 2)
        assert [b.shape for b in blocks] == [(2, 4), (2, 6)]
        assert torch.equal(blocks[0][0], wavs[0])
        assert torch.equal(blocks[0][1], torch.zeros(4))
        assert torch.equal(blocks[1][1], wavs[1])
        assert torch.equal(blocks[1][0], torch.zeros(6))


# --------------------------------------------------------------------------- #
# End-to-end infer stage
# --------------------------------------------------------------------------- #
def _external_config(testset, inference_dir, **overrides):
    cfg = {
        "inference_dir": str(inference_dir),
        "test_name": "valid",
        "mode": MODE,
        "device": "cpu",
        "ckpt": None,
        "use_ema": True,
        "testset": {
            "root": str(testset["testset_root"]),
            "librispeech_root": str(testset["librispeech_root"]),
            "num_channels": 2,
        },
        "duration": {"scale": 1.0, "speed": 1.0},
        "selection": {
            "min_duration": None,
            "max_duration": None,
            "num_dialogues": None,
            "seed": 0,
        },
        "sampling": {
            "steps": 2,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "seed": 0,
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


@pytest.fixture
def tiny_model(testset):
    return build_tiny(testset["vocab"])


class TestExternalInfer:
    def _run(self, testset, tiny_model, inference_dir, **overrides):
        cfg = _external_config(testset, inference_dir, **overrides)
        stats = run_external_inference(
            cfg,
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        return inference_dir / "valid", stats, cfg

    def test_output_contract_matches_the_sssd_layout(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, stats, _ = self._run(testset, tiny_model, tmp_path / "infer")
        assert stats == {
            "n_selected": 2,
            "n_skipped": 0,
            "n_not_sampled": 0,
            "n_other_shards": 0,
            # No `batching` block in the config -> every dialogue is its own
            # batch, the bit-exact sequential behaviour.
            "n_batches": 2,
        }

        assert (test_dir / "meta.scp").read_text("utf-8").splitlines() == [
            "000 meta/000.json",
            "001 meta/001.json",
        ]
        for name in ("wav.scp", "prompt.scp", "text.scp", "mix.scp"):
            assert (test_dir / name).is_file()
        # No ground-truth anchor exists for this test set.
        assert not (test_dir / "gt").exists()

        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["mode"] == MODE
        assert meta["has_reference_audio"] is False
        assert meta["turn_times"] == "ordinal"
        assert meta["testset"] == "covomix2-dialogue-testset"
        assert set(meta["channels"][0]) == {"gen_wav", "prompt_wav", "ref_text"}
        # The metric battery reads these keys; keep them present and typed.
        assert meta["num_channels"] == 2
        assert meta["sample_rate"] == FS
        assert isinstance(meta["rtf"], float)

    def test_duration_policy_is_recorded_in_every_meta(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            duration={"scale": 1.5, "speed": 2.0},
        )
        for key in ("000", "001"):
            meta = json.loads((test_dir / f"meta/{key}.json").read_text("utf-8"))
            assert meta["duration"]["duration_scale"] == 1.5
            assert meta["duration"]["speed"] == 2.0
            assert meta["duration"]["rule"] == "f5_prompt_ratio_per_speaker"
            assert meta["duration"]["predicted_sec"] > 0

    def test_generated_length_follows_the_predicted_duration(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer")
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        predicted = meta["duration"]["predicted_sec"]
        expected_frames = max(1, round(predicted * FS / HOP))

        data, sr = _read_wav(test_dir / "wav/000_ch0.wav")
        assert sr == FS
        assert data.shape[0] == expected_frames * HOP
        assert meta["window_duration_sec"] == pytest.approx(expected_frames * HOP / FS)

    def test_halving_speed_lengthens_the_generated_audio(
        self, testset, tiny_model, tmp_path
    ):
        fast, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "fast",
            duration={"scale": 1.0, "speed": 2.0},
        )
        slow, _, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "slow",
            duration={"scale": 1.0, "speed": 1.0},
        )
        fast_wav, _ = _read_wav(fast / "wav/000_ch0.wav")
        slow_wav, _ = _read_wav(slow / "wav/000_ch0.wav")
        assert slow_wav.shape[0] > fast_wav.shape[0]

    def test_prompt_wav_is_the_channels_own_utterance(
        self, testset, tiny_model, tmp_path
    ):
        # SIM-o embeds channels[k].prompt_wav; if it were the whole
        # full-width block, channel 1's reference would be silence.
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer")
        for ch in range(2):
            data, sr = _read_wav(test_dir / f"prompt/000_ch{ch}.wav")
            assert sr == FS
            assert float(abs(data).max()) > 0.0

    def test_reference_text_is_that_channels_turns_in_order(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer")
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        for ch in range(2):
            expected = " ".join(t["text"] for t in meta["turns"] if t["channel"] == ch)
            assert meta["channels"][ch]["ref_text"] == expected

    def test_branch_text_prepends_prompt_turns_in_channel_order(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, _, _ = self._run(testset, tiny_model, tmp_path / "infer")
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        prompt_turns = [
            Turn(p["channel"], "p", p["text"], 0.0, 0.0)
            for p in meta["prompt"]["turns"]
        ]
        dialogue_turns = [
            Turn(t["channel"], "d", t["text"], t["start"], t["end"])
            for t in meta["turns"]
        ]
        assert [t.channel for t in prompt_turns] == [0, 1]
        # The conditioning text the stage builds must equal prompt-then-
        # dialogue under the shared masking scheme.
        branches = build_branch_texts(prompt_turns + dialogue_turns, 2)
        assert len(branches) == 2
        assert branches[0][0] == branches[1][0]  # both open with <turn>

    def test_max_duration_band_excludes_and_reports(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, stats, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            selection={
                "min_duration": None,
                "max_duration": 0.5,  # below both fixture dialogues
                "num_dialogues": None,
                "seed": 0,
            },
        )
        assert stats["n_selected"] == 0
        assert stats["n_skipped"] == 2

    def test_wrong_mode_is_rejected(self, testset, tmp_path):
        cfg = _external_config(testset, tmp_path / "infer", mode="generate")
        with pytest.raises(ValueError, match="expected mode"):
            run_external_inference(cfg, training_config=testset["training_config"])

    def test_batched_run_keeps_the_per_dialogue_contract(
        self, testset, tiny_model, tmp_path
    ):
        # Both fixture dialogues share ONE ODE call; every per-dialogue
        # output (length, meta keys, layout) must be indistinguishable from
        # the sequential run's contract.
        test_dir, stats, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            batching={"max_batch_audio_sec": 10000.0, "max_batch_dialogues": 8},
        )
        assert stats["n_selected"] == 2
        assert stats["n_batches"] == 1
        for key in ("000", "001"):
            meta = json.loads((test_dir / f"meta/{key}.json").read_text("utf-8"))
            expected_frames = max(
                1, round(meta["duration"]["predicted_sec"] * FS / HOP)
            )
            for ch in range(2):
                data, sr = _read_wav(test_dir / f"wav/{key}_ch{ch}.wav")
                assert sr == FS
                # Batch padding must never leak into a dialogue's output:
                # each is vocoded at its own exact length.
                assert data.shape[0] == expected_frames * HOP
            assert meta["compute"]["batch_size"] == 2
            assert meta["compute"]["batch_id"] == 0
            assert isinstance(meta["rtf"], float)

    def test_dialogue_cap_splits_batches(self, testset, tiny_model, tmp_path):
        _, stats, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            batching={"max_batch_audio_sec": 10000.0, "max_batch_dialogues": 1},
        )
        assert stats["n_batches"] == 2

    def test_autocast_dtype_is_applied_and_recorded(
        self, testset, tiny_model, tmp_path
    ):
        test_dir, stats, _ = self._run(
            testset,
            tiny_model,
            tmp_path / "infer",
            sampling={
                "steps": 2,
                "cfg_strength": 2.0,
                "sway_sampling_coef": -1.0,
                "seed": 0,
                "autocast_dtype": "bfloat16",
            },
        )
        assert stats["n_selected"] == 2
        meta = json.loads((test_dir / "meta/000.json").read_text("utf-8"))
        assert meta["compute"]["autocast_dtype"] == "bfloat16"
        data, _ = _read_wav(test_dir / "wav/000_ch0.wav")
        assert data.shape[0] > 0


# --------------------------------------------------------------------------- #
# infer -> measure, end to end
# --------------------------------------------------------------------------- #
def _external_metrics_config(inference_dir: Path):
    """The real metric classes with the SSSD e2e suite's fake backends, but
    the no-reference interaction variant - the whole point being that this
    battery is otherwise UNCHANGED from conf/metrics.yaml."""
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": "valid"}]},
            "metrics": [
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.asr."
                            "ConversationASRMetric"
                        ),
                        "transcriber": _fake("FakeTranscriber"),
                        "normalizer": _fake("FakeNormalizer"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.speaker."
                            "SpeakerSimilarityMetric"
                        ),
                        "embedder": _fake("FakeEmbedder"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.quality."
                            "QualityMetric"
                        ),
                        "mos_backend": _fake("FakeMOSBackend"),
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics."
                            "interaction_no_reference."
                            "NoReferenceInteractionMetric"
                        ),
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
            ],
        }
    )


class TestExternalMeasure:
    def test_full_battery_runs_on_external_output(self, testset, tiny_model, tmp_path):
        inference_dir = tmp_path / "infer"
        run_external_inference(
            _external_config(testset, inference_dir),
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        results = measure(_external_metrics_config(inference_dir))

        # Every metric class the SSSD battery uses produces its documented
        # summary keys here too - the measure stage needed no changes.
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

    def test_dur_w1_keys_are_null_without_a_reference(
        self, testset, tiny_model, tmp_path
    ):
        inference_dir = tmp_path / "infer"
        run_external_inference(
            _external_config(testset, inference_dir),
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )
        results = measure(_external_metrics_config(inference_dir))
        key = next(k for k in results if k.endswith("NoReferenceInteractionMetric"))
        summary = results[key]["valid"]
        # Distances against a ground truth that does not exist must be
        # null, never a fabricated 0.0 (which would read as "perfect").
        for event in ("ipu", "pause", "gap", "overlap"):
            assert summary[f"{event}_dur_w1"] is None
        # The count/duration keys are still computed.
        assert summary["ipu_per_min"] is not None

    def test_no_reference_variant_rejects_meta_that_has_ground_truth(self):
        metric = NoReferenceInteractionMetric()
        meta = {
            "window_id": "w0",
            "window_duration_sec": 1.0,
            "channels": [{"gen_wav": "a.wav", "gt_wav": "b.wav"}],
        }
        with pytest.raises(ValueError, match="carries gt_wav"):
            metric._score_window(meta, Path("."))


def test_system_dispatch_literal_matches_mode():
    """``system.py`` compares against a LITERAL rather than importing MODE,
    so that dispatching never pulls the external modules into an SSSD run.
    This pins the two equal so they cannot drift apart silently."""
    from egs3.conversational.tts.src.system import EXTERNAL_MODE

    assert EXTERNAL_MODE == MODE


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #
class TestAssignShard:
    def test_shards_partition_the_input_exactly(self):
        durations = [float(i % 7 + 1) for i in range(50)]
        indices = list(range(50))
        union = []
        for s in range(4):
            union.extend(assign_shard(indices, durations, s, 4))
        # Exact partition: every dialogue generated once and only once.
        # Overlap would double-count in every pooled metric; a gap would
        # silently shrink the evaluated set.
        assert sorted(union) == indices

    def test_single_shard_is_the_identity(self):
        indices = [3, 1, 4, 1, 5]
        assert assign_shard(indices, [1.0] * 10, 0, 1) == indices

    def test_output_is_sorted_and_deterministic(self):
        durations = [float((i * 37) % 91) for i in range(60)]
        first = assign_shard(list(range(60)), durations, 2, 5)
        second = assign_shard(list(range(60)), durations, 2, 5)
        assert first == second == sorted(first)

    def test_long_tail_is_spread_not_concentrated(self):
        # Four very long dialogues among many short ones. Striping by
        # index % 4 would put all four (indices 0, 4, 8, 12) in shard 0;
        # length balancing must not.
        durations = [200.0 if i % 4 == 0 and i < 16 else 5.0 for i in range(40)]
        loads = [
            sum(durations[i] for i in assign_shard(list(range(40)), durations, s, 4))
            for s in range(4)
        ]
        assert max(loads) / min(loads) < 1.2
        striped = sum(durations[i] for i in range(40) if i % 4 == 0)
        assert max(loads) < striped  # strictly better than striping

    def test_balanced_within_a_modest_factor_on_skewed_lengths(self):
        # Shape of the real set: a heavy tail carrying much of the audio.
        durations = [200.0] * 16 + [30.0] * 84
        loads = [
            sum(durations[i] for i in assign_shard(list(range(100)), durations, s, 4))
            for s in range(4)
        ]
        assert max(loads) / min(loads) < 1.1

    def test_invalid_shard_spec_raises(self):
        with pytest.raises(ValueError, match="shard_count"):
            assign_shard([0], [1.0], 0, 0)
        with pytest.raises(ValueError, match="shard_index"):
            assign_shard([0], [1.0], 4, 4)


class TestPlanBatches:
    def test_none_budgets_mean_singletons_in_input_order(self):
        # The bit-exact sequential behaviour: no sorting, no packing.
        assert plan_batches([3, 1, 2], [1.0, 2.0, 3.0, 4.0]) == [[3], [1], [2]]

    def test_budget_packs_length_sorted_dialogues(self):
        batches = plan_batches(
            [0, 1, 2, 3], [10.0, 9.0, 5.0, 4.0], max_batch_audio_sec=20.0
        )
        assert batches == [[0, 1], [2, 3]]

    def test_cost_is_padded_audio_not_the_sum(self):
        # Pairing 10s with 1s costs 2*10=20 of PADDED audio, over a 15s
        # budget - even though the sum of true lengths (11s) is under it.
        assert plan_batches([0, 1], [10.0, 1.0], max_batch_audio_sec=15.0) == [
            [0],
            [1],
        ]

    def test_over_budget_dialogue_still_gets_a_singleton(self):
        # The budget bounds batch growth; it never excludes work.
        assert plan_batches([0, 1], [100.0, 1.0], max_batch_audio_sec=10.0) == [
            [0],
            [1],
        ]

    def test_dialogue_cap_applies_independently(self):
        batches = plan_batches(
            list(range(5)),
            [1.0] * 5,
            max_batch_audio_sec=100.0,
            max_batch_dialogues=2,
        )
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_plan_is_independent_of_input_order(self):
        a = plan_batches([4, 2, 0, 3, 1], [5.0] * 5, max_batch_audio_sec=10.0)
        b = plan_batches([0, 1, 2, 3, 4], [5.0] * 5, max_batch_audio_sec=10.0)
        assert a == b == [[0, 1], [2, 3], [4]]

    def test_invalid_budgets_raise(self):
        with pytest.raises(ValueError, match="max_batch_audio_sec"):
            plan_batches([0], [1.0], max_batch_audio_sec=0.0)
        with pytest.raises(ValueError, match="max_batch_dialogues"):
            plan_batches([0], [1.0], max_batch_dialogues=0)


class TestShardedBatchPlan:
    def test_shards_take_whole_batches_and_union_is_the_full_plan(self):
        durations = [float(i) for i in range(20)]  # 17,18,19 out of band
        base = OmegaConf.create({"max_duration": 16.5, "num_dialogues": 8, "seed": 0})
        selected, counts = select_records(
            [_record(["abc"], ["abc", "de"])] * 20, durations, base
        )
        assert counts == {"n_out_of_band": 3, "n_not_sampled": 9}

        # The batch plan is computed over the FULL selection; shards take
        # whole batches, so the union of all shards is exactly the plan and
        # batch composition never depends on shard_count.
        batches = plan_batches(selected, durations, max_batch_audio_sec=30.0)
        costs = [len(b) * max(durations[i] for i in b) for b in batches]
        union = []
        for s in range(3):
            for b in assign_shard(list(range(len(batches))), costs, s, 3):
                union.extend(batches[b])
        assert sorted(union) == selected


class TestShardedInfer:
    def _cfg(self, testset, inference_dir, shard_index, shard_count):
        return _external_config(
            testset,
            inference_dir,
            selection={
                "min_duration": None,
                "max_duration": None,
                "num_dialogues": None,
                "seed": 0,
                "shard_index": shard_index,
                "shard_count": shard_count,
            },
        )

    def _run_shard(self, testset, tiny_model, inference_dir, index, count):
        return run_external_inference(
            self._cfg(testset, inference_dir, index, count),
            training_config=testset["training_config"],
            model=tiny_model,
            vocoder=FakeVocoder(),
        )

    def test_shards_write_own_scps_and_merge_to_the_full_set(
        self, testset, tiny_model, tmp_path
    ):
        from egs3.conversational.tts.local.merge_shards import merge

        inference_dir = tmp_path / "infer"
        for s in range(2):
            self._run_shard(testset, tiny_model, inference_dir, s, 2)

        test_dir = inference_dir / "valid"
        # Each shard wrote its own SCP; no plain meta.scp exists yet.
        assert not (test_dir / "meta.scp").exists()
        assert (test_dir / "meta.scp.0of2").is_file()
        assert (test_dir / "meta.scp.1of2").is_file()

        written = merge(test_dir)
        assert written["meta"] == 2  # the fixture has two dialogues
        ids = [
            line.split(" ", 1)[0]
            for line in (test_dir / "meta.scp").read_text("utf-8").splitlines()
        ]
        assert sorted(ids) == ["000", "001"]

    def test_merge_refuses_a_partial_run(self, testset, tiny_model, tmp_path):
        from egs3.conversational.tts.local.merge_shards import merge

        inference_dir = tmp_path / "infer"
        self._run_shard(testset, tiny_model, inference_dir, 0, 2)  # shard 1 absent

        # A silently short meta.scp would score a subset while looking like a
        # complete run - the exact failure this recipe tries not to have.
        with pytest.raises(SystemExit, match="Refusing to write a partial merge"):
            merge(inference_dir / "valid")
        assert not (inference_dir / "valid" / "meta.scp").exists()

    def test_merge_allows_partial_when_asked(self, testset, tiny_model, tmp_path):
        from egs3.conversational.tts.local.merge_shards import merge

        inference_dir = tmp_path / "infer"
        self._run_shard(testset, tiny_model, inference_dir, 0, 2)
        written = merge(inference_dir / "valid", allow_partial=True)
        assert written["meta"] >= 1

    def test_unsharded_run_writes_plain_names(self, testset, tiny_model, tmp_path):
        inference_dir = tmp_path / "infer"
        self._run_shard(testset, tiny_model, inference_dir, 0, 1)
        test_dir = inference_dir / "valid"
        assert (test_dir / "meta.scp").is_file()
        assert not list(test_dir.glob("meta.scp.*"))

    def test_sharded_union_is_bit_identical_to_unsharded(
        self, testset, tiny_model, tmp_path
    ):
        # The core reproducibility promise of shard-over-batches: batch
        # composition and every noise draw are a function of the config
        # only, so shard_count never changes a single generated sample.
        unsharded_dir = tmp_path / "unsharded"
        self._run_shard(testset, tiny_model, unsharded_dir, 0, 1)
        sharded_dir = tmp_path / "sharded"
        for s in range(2):
            self._run_shard(testset, tiny_model, sharded_dir, s, 2)
        wavs = sorted((unsharded_dir / "valid" / "wav").glob("*.wav"))
        assert wavs  # the fixture generates something
        for wav in wavs:
            sharded = sharded_dir / "valid" / "wav" / wav.name
            assert sharded.read_bytes() == wav.read_bytes()
