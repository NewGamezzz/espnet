"""Tests for the training-style external dialogue manifest loader
(``src/external_testset.py::load_external_manifest``).

The manifest is the corpus interface for external dialogue test sets that
ship their own prompts (and optionally per-channel ground-truth audio), e.g.
ZipVoice-Dialog test-en.  One JSONL line per dialogue, shaped like the
training ``WindowRecord``: explicit ``num_channels``, turns with explicit
``channel`` indices (no alternation rule), and one prompt per channel.

Fixture-based and CPU-only: fabricated wavs, the conftest vocab.  Text in
the fixtures is restricted to the conftest charset (space plus ``a``-``j``)
because the loader normalizes against the extended vocab.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from egs3.conversational.tts.dataset.preprocessing.text import (
    normalize_text,
    vocab_charset,
)
from egs3.conversational.tts.src.chunked_inference import MODE as CHUNKED_MODE
from egs3.conversational.tts.src.chunked_inference import (
    run_chunked_inference,
)
from egs3.conversational.tts.src.external_anchor import MODE as GT_ANCHOR_MODE
from egs3.conversational.tts.src.external_anchor import (
    run_external_gt,
)
from egs3.conversational.tts.src.external_testset import (
    ExternalRecord,
    load_external_manifest,
)
from egs3.conversational.tts.src.system import GT_ANCHOR_MODE as SYSTEM_GT_ANCHOR_MODE
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
from .test_inference import HOP, FakeVocoder, _read_wav

FS = 24000


def _write_wav(path: Path, seconds: float, sr: int = FS, freq: float = 400.0) -> None:
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    data = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


# (window_id, session_id, turns[(channel, text)], prompts[(text, sec)], gt_secs|None)
TWO_SPK = (
    "d2",
    "s",
    [(0, "abc"), (1, "def"), (0, "gab")],
    [("abc", 2.0), ("de", 1.5)],
    [4.0, 4.0],
)
ONE_SPK = ("d1", "s", [(0, "cab"), (0, "bad")], [("fed", 1.0)], [3.0])
NO_GT = ("d0", "s", [(0, "abc"), (1, "def")], [("abc", 2.0), ("de", 1.5)], None)


def write_manifest(tmp_path: Path, specs, name: str = "zipvoice") -> dict:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for wid, sid, turns, prompts, gt in specs:
        channels = []
        for ch, (ptext, psec) in enumerate(prompts):
            prel = f"prompt/{wid}_ch{ch}.wav"
            _write_wav(root / prel, psec)
            entry = {"prompt_wav": prel, "prompt_text": ptext}
            if gt is not None:
                grel = f"gt/{wid}_ch{ch}.wav"
                _write_wav(root / grel, gt[ch], freq=300.0 * (ch + 1))
                entry["gt_wav"] = grel
            channels.append(entry)
        lines.append(
            {
                "window_id": wid,
                "session_id": sid,
                "num_channels": len(prompts),
                "turns": [
                    {"channel": ch, "speaker": f"S{ch + 1}", "text": text}
                    for ch, text in turns
                ],
                "channels": channels,
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
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
        "root": root,
        "manifest": manifest,
        "vocab": vocab,
        "lines": lines,
        "training_config": training_config,
    }


@pytest.fixture
def manifest(tmp_path):
    return write_manifest(tmp_path, [TWO_SPK, ONE_SPK, NO_GT])


def _load(fx) -> list[ExternalRecord]:
    return load_external_manifest(fx["manifest"], fx["vocab"])


class TestLoadExternalManifest:
    def test_channel_count_is_per_dialogue(self, manifest):
        records = _load(manifest)
        assert [r.dialogue_id for r in records] == ["d2", "d1", "d0"]
        assert [r.num_channels for r in records] == [2, 1, 2]
        one = records[1]
        assert [p.channel for p in one.prompts] == [0]
        assert [t.channel for t in one.turns] == [0, 0]

    def test_turn_channels_come_from_the_manifest_not_alternation(self, manifest):
        two = _load(manifest)[0]
        # [S1, S2, S1] as written - and a same-speaker pair is kept as two
        # turns, never merged and never re-alternated.
        assert [t.channel for t in two.turns] == [0, 1, 0]
        fx = write_manifest(
            manifest["root"].parent / "again",
            [
                (
                    "x",
                    "s",
                    [(0, "abc"), (0, "def"), (1, "gab")],
                    [("a", 1.0), ("b", 1.0)],
                    None,
                )
            ],
        )
        assert [t.channel for t in _load(fx)[0].turns] == [0, 0, 1]

    def test_text_is_normalized_against_the_extended_vocab(self, manifest):
        charset = vocab_charset(EXT_TOKENS)
        two = _load(manifest)[0]
        assert [t.text for t in two.turns] == [
            normalize_text(text, charset) for _, text in TWO_SPK[2]
        ]
        assert two.prompts[1].text == normalize_text("de", charset)

    def test_turn_times_are_ordinals(self, manifest):
        two = _load(manifest)[0]
        assert [t.start for t in two.turns] == [0.0, 1.0, 2.0]
        assert sorted(two.turns, key=lambda t: t.start) == two.turns

    def test_paths_resolve_against_the_manifest_directory(self, manifest):
        two = _load(manifest)[0]
        assert two.prompts[0].audio_path == manifest["root"] / "prompt/d2_ch0.wav"
        assert two.prompts[0].audio_path.is_absolute()

    def test_ground_truth_is_carried_when_present(self, manifest):
        records = _load(manifest)
        two, one, none = records
        assert two.gt_paths == (
            manifest["root"] / "gt/d2_ch0.wav",
            manifest["root"] / "gt/d2_ch1.wav",
        )
        assert two.gt_duration_sec == pytest.approx(4.0)
        assert one.gt_paths == (manifest["root"] / "gt/d1_ch0.wav",)
        assert one.gt_duration_sec == pytest.approx(3.0)
        assert none.gt_paths is None
        assert none.gt_duration_sec is None

    def test_channel_chars_follow_the_explicit_channels(self, manifest):
        two = _load(manifest)[0]
        charset = vocab_charset(EXT_TOKENS)
        n = lambda s: len(normalize_text(s, charset).encode("utf-8"))  # noqa: E731
        assert two.channel_chars == [n("abc") + n("gab"), n("def")]


class TestManifestValidation:
    def test_channel_without_a_turn_raises(self, tmp_path):
        fx = write_manifest(
            tmp_path, [("x", "s", [(0, "abc")], [("a", 1.0), ("b", 1.0)], None)]
        )
        with pytest.raises(ValueError, match="no turn"):
            _load(fx)

    def test_turn_channel_out_of_range_raises(self, tmp_path):
        fx = write_manifest(
            tmp_path,
            [("x", "s", [(0, "abc"), (2, "def")], [("a", 1.0), ("b", 1.0)], None)],
        )
        with pytest.raises(ValueError, match="channel 2"):
            _load(fx)

    def test_num_channels_must_match_the_channel_list(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        line = dict(fx["lines"][0], num_channels=3)
        fx["manifest"].write_text(json.dumps(line) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="num_channels"):
            _load(fx)

    def test_missing_prompt_audio_raises(self, tmp_path):
        fx = write_manifest(tmp_path, [NO_GT])
        (fx["root"] / "prompt/d0_ch1.wav").unlink()
        with pytest.raises(FileNotFoundError, match="prompt audio"):
            _load(fx)

    def test_missing_ground_truth_audio_raises(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        (fx["root"] / "gt/d2_ch1.wav").unlink()
        with pytest.raises(FileNotFoundError, match="ground-truth audio"):
            _load(fx)

    def test_ground_truth_on_only_some_channels_raises(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        line = json.loads(fx["manifest"].read_text().splitlines()[0])
        del line["channels"][1]["gt_wav"]
        fx["manifest"].write_text(json.dumps(line) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="every channel or none"):
            _load(fx)

    def test_ground_truth_channels_must_share_a_duration(self, tmp_path):
        fx = write_manifest(tmp_path, [("x", "s", TWO_SPK[2], TWO_SPK[3], [4.0, 3.0])])
        with pytest.raises(ValueError, match="duration"):
            _load(fx)

    def test_duplicate_window_id_raises(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK])
        text = fx["manifest"].read_text()
        fx["manifest"].write_text(text + text, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            _load(fx)

    def test_empty_turn_after_normalization_raises(self, tmp_path):
        # "+" is outside the conftest charset, so the turn normalizes to "".
        fx = write_manifest(
            tmp_path,
            [("x", "s", [(0, "+"), (1, "abc")], [("a", 1.0), ("b", 1.0)], None)],
        )
        with pytest.raises(ValueError, match="empty text"):
            _load(fx)

    def test_empty_prompt_text_raises(self, tmp_path):
        fx = write_manifest(
            tmp_path,
            [("x", "s", [(0, "abc"), (1, "abc")], [("+", 1.0), ("b", 1.0)], None)],
        )
        with pytest.raises(ValueError, match="prompt text is empty"):
            _load(fx)


# --------------------------------------------------------------------------- #
# chunked infer driven by a manifest
# --------------------------------------------------------------------------- #
def _manifest_config(fx, inference_dir, chunk, **overrides):
    cfg = {
        "inference_dir": str(inference_dir),
        "test_name": "valid",
        "mode": CHUNKED_MODE,
        "device": "cpu",
        "ckpt": None,
        "use_ema": True,
        "testset": {"manifest": str(fx["manifest"]), "name": "zipvoice-dialog-test-en"},
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
        "chunk": chunk,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _run_chunked(fx, inference_dir, chunk, **overrides):
    cfg = _manifest_config(fx, inference_dir, chunk, **overrides)
    stats = run_chunked_inference(
        cfg,
        training_config=fx["training_config"],
        model=build_tiny(fx["vocab"]),
        vocoder=FakeVocoder(),
    )
    return inference_dir / "valid", stats, cfg


def _meta(test_dir, wid):
    return json.loads((test_dir / f"meta/{wid}.json").read_text("utf-8"))


class TestChunkedInferFromManifest:
    def test_records_come_from_the_manifest(self, manifest, tmp_path):
        test_dir, stats, _ = _run_chunked(manifest, tmp_path / "infer", {"turns": 2})
        assert stats["n_selected"] == 3
        assert (test_dir / "meta.scp").read_text("utf-8").splitlines() == [
            "d2 meta/d2.json",
            "d1 meta/d1.json",
            "d0 meta/d0.json",
        ]
        assert _meta(test_dir, "d2")["testset"] == "zipvoice-dialog-test-en"

    def test_single_speaker_dialogue_is_a_one_channel_record(self, manifest, tmp_path):
        test_dir, _, _ = _run_chunked(manifest, tmp_path / "infer", {"turns": 2})
        meta = _meta(test_dir, "d1")
        assert meta["num_channels"] == 1
        assert len(meta["channels"]) == 1
        assert [t["channel"] for t in meta["turns"]] == [0, 0]
        assert (test_dir / "wav/d1_ch0.wav").is_file()
        assert not (test_dir / "wav/d1_ch1.wav").exists()
        wav, _ = _read_wav(test_dir / "wav/d1_ch0.wav")
        mix, _ = _read_wav(test_dir / "mix/d1.wav")
        # A one-channel mixdown IS the channel (sum / 1).
        assert np.allclose(wav, mix)
        keys = [
            ln.split()[0]
            for ln in (test_dir / "wav.scp").read_text("utf-8").splitlines()
        ]
        assert [k for k in keys if k.startswith("d1_")] == ["d1_ch0"]

    def test_mixed_channel_counts_share_one_batch(self, manifest, tmp_path):
        # d2 (2 channels) and d1 (1 channel) packed into ONE ODE call: the
        # packed row layout takes per-item channel counts.
        test_dir, stats, _ = _run_chunked(
            manifest,
            tmp_path / "infer",
            {"turns": 10},
            batching={"max_batch_audio_sec": None, "max_batch_dialogues": 3},
        )
        assert stats["n_batches"] == 1
        assert _meta(test_dir, "d1")["chunking"]["chunks"][0]["batch_size"] == 3

    def test_ground_truth_is_written_into_the_meta(self, manifest, tmp_path):
        test_dir, _, _ = _run_chunked(manifest, tmp_path / "infer", {"turns": 2})
        meta = _meta(test_dir, "d2")
        assert meta["has_reference_audio"] is True
        assert meta["gt_duration_sec"] == pytest.approx(4.0)
        assert [c["gt_wav"] for c in meta["channels"]] == [
            "gt/d2_ch0.wav",
            "gt/d2_ch1.wav",
        ]
        for ch in range(2):
            gt, sr = _read_wav(test_dir / f"gt/d2_ch{ch}.wav")
            assert sr == FS
            assert gt.shape[0] == pytest.approx(4.0 * FS, abs=1)
        # gen wav and gt wav are different files: the metric reads both.
        assert (test_dir / "wav/d2_ch0.wav").is_file()
        assert (test_dir / "gt.scp").read_text("utf-8").splitlines()[:2] == [
            "d2_ch0 gt/d2_ch0.wav",
            "d2_ch1 gt/d2_ch1.wav",
        ]

    def test_dialogue_without_ground_truth_stays_reference_free(
        self, manifest, tmp_path
    ):
        test_dir, _, _ = _run_chunked(manifest, tmp_path / "infer", {"turns": 2})
        meta = _meta(test_dir, "d0")
        assert meta["has_reference_audio"] is False
        assert meta["gt_duration_sec"] is None
        assert all("gt_wav" not in c for c in meta["channels"])
        assert not (test_dir / "gt/d0_ch0.wav").exists()

    def test_predicted_duration_is_the_default_source(self, manifest, tmp_path):
        test_dir, _, _ = _run_chunked(manifest, tmp_path / "infer", {"turns": 2})
        meta = _meta(test_dir, "d2")
        assert meta["duration"]["source"] == "predicted"
        assert meta["duration"]["gt_sec"] == pytest.approx(4.0)
        assert meta["duration"]["predicted_over_gt"] == pytest.approx(
            meta["duration"]["predicted_sec"] / 4.0
        )
        assert _meta(test_dir, "d0")["duration"]["gt_sec"] is None
        assert _meta(test_dir, "d0")["duration"]["predicted_over_gt"] is None

    def test_ground_truth_duration_source_generates_the_gt_length(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        test_dir, _, _ = _run_chunked(
            fx,
            tmp_path / "infer",
            {"turns": 2},
            duration={"scale": 1.0, "speed": 1.0, "source": "ground_truth"},
        )
        for wid, gt_sec in (("d2", 4.0), ("d1", 3.0)):
            meta = _meta(test_dir, wid)
            assert meta["duration"]["source"] == "ground_truth"
            chunks = meta["chunking"]["chunks"]
            total = sum(c["predicted_sec"] for c in chunks)
            assert total == pytest.approx(gt_sec, abs=1e-6)
            # Frame rounding only: the generated wave is the GT length.
            assert meta["window_duration_sec"] == pytest.approx(gt_sec, abs=HOP / FS)
            # The rule's own estimate is still recorded, so the ratio is
            # readable even on the oracle arm.
            assert meta["duration"]["predicted_sec"] > 0
            assert meta["duration"]["predicted_over_gt"] == pytest.approx(
                meta["duration"]["predicted_sec"] / gt_sec
            )

    def test_ground_truth_source_keeps_the_chunk_proportions(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        pred_dir, _, _ = _run_chunked(fx, tmp_path / "pred", {"turns": 1})
        gt_dir, _, _ = _run_chunked(
            fx,
            tmp_path / "gtdur",
            {"turns": 1},
            duration={"scale": 1.0, "speed": 1.0, "source": "ground_truth"},
        )
        pred = [c["predicted_sec"] for c in _meta(pred_dir, "d2")["chunking"]["chunks"]]
        gt = [c["predicted_sec"] for c in _meta(gt_dir, "d2")["chunking"]["chunks"]]
        factor = 4.0 / sum(pred)
        assert gt == pytest.approx([p * factor for p in pred])

    def test_ground_truth_source_without_ground_truth_raises(self, tmp_path):
        # One reference-free dialogue poisons the oracle arm as a whole: the
        # arm is "generate the reference length", so it must not exist.
        fx = write_manifest(tmp_path, [TWO_SPK, NO_GT])
        with pytest.raises(ValueError, match="no ground-truth"):
            _run_chunked(
                fx,
                tmp_path / "infer",
                {"turns": 2},
                duration={"scale": 1.0, "speed": 1.0, "source": "ground_truth"},
            )

    def test_unknown_duration_source_raises(self, manifest, tmp_path):
        with pytest.raises(ValueError, match="duration.source"):
            _run_chunked(
                manifest,
                tmp_path / "infer",
                {"turns": 2},
                duration={"scale": 1.0, "speed": 1.0, "source": "oracle"},
            )

    def test_manifest_and_root_are_mutually_exclusive(self, manifest, tmp_path):
        with pytest.raises(ValueError, match="testset.manifest"):
            _run_chunked(
                manifest,
                tmp_path / "infer",
                {"turns": 2},
                testset={"manifest": str(manifest["manifest"]), "root": "x"},
            )


# --------------------------------------------------------------------------- #
# measure over a manifest run: the parent InteractionMetric (W1 vs GT live)
# --------------------------------------------------------------------------- #
def _metrics_config(inference_dir: Path):
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": "valid"}]},
            "metrics": [
                {
                    "metric": {
                        "_target_": "egs3.conversational.tts.src.metrics.asr.ConversationASRMetric",
                        "transcriber": _fake("FakeTranscriber"),
                        "normalizer": _fake("FakeNormalizer"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": "egs3.conversational.tts.src.metrics.speaker.SpeakerSimilarityMetric",
                        "embedder": _fake("FakeEmbedder"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": "egs3.conversational.tts.src.metrics.quality.QualityMetric",
                        "mos_backend": _fake("FakeMOSBackend"),
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": "egs3.conversational.tts.src.metrics.interaction.InteractionMetric",
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
            ],
        }
    )


def _summary(results) -> dict:
    """Flatten measure() output ({metric_class: {test_name: summary}})."""
    out = {}
    for per_test in results.values():
        out.update(per_test["valid"])
    return out


class TestMeasureFromManifest:
    def test_full_battery_with_reference_w1(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        test_dir, _, _ = _run_chunked(fx, tmp_path / "infer", {"turns": 2})
        summary = _summary(measure(_metrics_config(tmp_path / "infer")))
        keys = ASR_SUMMARY_KEYS | SPEAKER_SUMMARY_KEYS | QUALITY_SUMMARY_KEYS
        assert keys <= set(summary)
        assert INTERACTION_SUMMARY_KEYS <= set(summary)
        # Fake VAD: one IPU per channel over the whole wav, gen and gt alike,
        # so the ipu W1 is |gen_len - gt_len| pooled - defined, not None.
        assert summary["ipu_dur_w1"] is not None


# --------------------------------------------------------------------------- #
# gt anchor mode
# --------------------------------------------------------------------------- #
def _anchor_config(fx, inference_dir, **overrides):
    cfg = {
        "inference_dir": str(inference_dir),
        "test_name": "valid",
        "mode": GT_ANCHOR_MODE,
        "device": "cpu",
        "testset": {"manifest": str(fx["manifest"]), "name": "zipvoice-dialog-test-en"},
        "selection": {"dialogue_ids": None},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


class TestGtAnchor:
    def test_dispatch_literal_matches_mode(self):
        assert SYSTEM_GT_ANCHOR_MODE == GT_ANCHOR_MODE == "generate_external_gt"

    def test_gen_is_the_ground_truth(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        stats = run_external_gt(
            _anchor_config(fx, tmp_path / "anchor"),
            training_config=fx["training_config"],
        )
        test_dir = tmp_path / "anchor" / "valid"
        assert stats == {"n_selected": 2, "n_skipped": 0}
        meta = _meta(test_dir, "d2")
        assert meta["mode"] == GT_ANCHOR_MODE
        assert meta["has_reference_audio"] is True
        assert meta["num_channels"] == 2
        assert meta["window_duration_sec"] == pytest.approx(4.0, abs=1 / FS)
        assert meta["duration"]["source"] == "ground_truth"
        assert meta["turn_times"] == "ordinal"
        for ch in range(2):
            gen, sr = _read_wav(test_dir / meta["channels"][ch]["gen_wav"])
            gt, _ = _read_wav(test_dir / meta["channels"][ch]["gt_wav"])
            assert sr == FS
            assert np.array_equal(gen, gt)
            assert (test_dir / meta["channels"][ch]["prompt_wav"]).is_file()
            assert meta["channels"][ch]["ref_text"]
        mix, _ = _read_wav(test_dir / meta["mix_wav"])
        assert mix.shape[0] == gen.shape[0]
        one = _meta(test_dir, "d1")
        assert one["num_channels"] == 1
        for name in (
            "meta.scp",
            "wav.scp",
            "prompt.scp",
            "text.scp",
            "mix.scp",
            "gt.scp",
        ):
            assert (test_dir / name).is_file()

    def test_w1_collapses_to_zero_on_the_anchor(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        run_external_gt(
            _anchor_config(fx, tmp_path / "anchor"),
            training_config=fx["training_config"],
        )
        summary = _summary(measure(_metrics_config(tmp_path / "anchor")))
        for event in ("ipu", "pause", "gap", "overlap"):
            w1 = summary[f"{event}_dur_w1"]
            assert w1 is None or w1 == pytest.approx(0.0, abs=1e-9)
        assert summary["ipu_dur_w1"] == pytest.approx(0.0, abs=1e-9)

    def test_dialogue_without_ground_truth_is_rejected(self, tmp_path):
        fx = write_manifest(tmp_path, [NO_GT])
        with pytest.raises(ValueError, match="no ground-truth"):
            run_external_gt(
                _anchor_config(fx, tmp_path / "anchor"),
                training_config=fx["training_config"],
            )

    def test_pinned_ids_select_the_subset(self, tmp_path):
        fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
        ids = tmp_path / "ids.txt"
        ids.write_text("d1\n", encoding="utf-8")
        stats = run_external_gt(
            _anchor_config(
                fx, tmp_path / "anchor", selection={"dialogue_ids": str(ids)}
            ),
            training_config=fx["training_config"],
        )
        assert stats == {"n_selected": 1, "n_skipped": 1}
        assert (tmp_path / "anchor/valid/meta.scp").read_text(
            "utf-8"
        ) == "d1 meta/d1.json\n"
