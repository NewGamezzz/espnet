"""End-to-end tests for the builder CLI over the fabricated 2-session corpus.

Covers: output file layout, dataset.json/dialogues.jsonl 1:1 sorted-samples
invariant, determinism across two runs (byte-identical modulo the absolute
out-dir prefix), the printed stats block, and metadata-JSON vs heuristic
gender_source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from dataset.builder import (
    _select_measurement_turns,
    build,
    load_config,
    main,
    resolve_dataset_root,
    split_sessions,
)
from dataset.preprocessing.sssd import Turn

VARIANTS = ("tac", "mono")
SPLITS = ("train", "valid", "test")


def _load_variant_split(out_dir, variant, split):
    d = out_dir / variant / split
    dialogues = [
        json.loads(line) for line in (d / "dialogues.jsonl").read_text().splitlines()
    ]
    dataset_json = json.loads((d / "dataset.json").read_text())
    return dialogues, dataset_json


class TestBuildEndToEnd:
    def test_output_layout_and_sorted_samples_invariant(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path
    ):
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)

        assert (out_dir / "audio" / "train").is_dir()

        for variant in VARIANTS:
            for split in SPLITS:
                dialogues, dataset_json = _load_variant_split(out_dir, variant, split)
                example_ids = [r["example_id"] for r in dialogues]
                assert set(dataset_json["samples"]) == set(example_ids)
                assert dataset_json["samples"] == sorted(dataset_json["samples"])
                assert len(example_ids) == len(set(example_ids)), "duplicate example_id"
                entry = dataset_json["data_entry"]
                assert entry == [
                    {
                        "name": "dialogue",
                        "path": str(
                            (out_dir / variant / split / "dialogues.jsonl").resolve()
                        ),
                        "reader": "dialogue",
                    }
                ]

        # Both fixture sessions land in train under the real split ratios.
        train_mono, _ = _load_variant_split(out_dir, "mono", "train")
        assert train_mono, "expected mono records in train split"
        train_tac, _ = _load_variant_split(out_dir, "tac", "train")
        assert (
            train_tac
        ), "sessA's tight-alternation turns must yield TAC-eligible windows"
        assert (
            len(train_tac) % 2 == 0
        ), "TAC records always come in same-window channel pairs"

        valid_mono, _ = _load_variant_split(out_dir, "mono", "valid")
        test_mono, _ = _load_variant_split(out_dir, "mono", "test")
        assert valid_mono == []
        assert test_mono == []

    def test_tac_dropped_windows_are_mono_only(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path
    ):
        """sessB opens with a single-speaker stretch: at least one window must
        be TAC-ineligible (mono-only), so mono record count exceeds half the
        tac record count."""
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)
        train_mono, _ = _load_variant_split(out_dir, "mono", "train")
        train_tac, _ = _load_variant_split(out_dir, "tac", "train")
        assert len(train_mono) > len(train_tac) // 2

    def test_audio_paths_are_absolute_and_exist(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path
    ):
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)
        dialogues, _ = _load_variant_split(out_dir, "mono", "train")
        for rec in dialogues:
            audio_msg = rec["messages"][3]
            assert audio_msg[0] == "assistant" and audio_msg[1] == "audio"
            p = Path(audio_msg[2])
            assert p.is_absolute()
            assert p.is_file()

    def test_stats_block_prints(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path, capsys
    ):
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)
        out = capsys.readouterr().out
        assert "windows" in out
        assert "tac-dropped" in out
        assert "gender_source" in out
        assert "caption length" in out

    def test_determinism_byte_identical_modulo_out_dir(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path
    ):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        build(bagpiper_corpus["root"], dir_a, tiny_builder_cfg, seed=0)
        build(bagpiper_corpus["root"], dir_b, tiny_builder_cfg, seed=0)

        for variant in VARIANTS:
            for split in SPLITS:
                path_a = dir_a / variant / split / "dialogues.jsonl"
                path_b = dir_b / variant / split / "dialogues.jsonl"
                text_a = path_a.read_text().replace(str(dir_a.resolve()), "<OUT>")
                text_b = path_b.read_text().replace(str(dir_b.resolve()), "<OUT>")
                assert text_a == text_b, f"{variant}/{split} not deterministic"

    def test_metadata_json_respected_vs_absent(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path, capsys
    ):
        metadata = {
            "sessA_spk0": {"gender": "male"},
            "sessA_spk1": {"gender": "female"},
        }
        out_dir = tmp_path / "with_meta"
        build(
            bagpiper_corpus["root"],
            out_dir,
            tiny_builder_cfg,
            seed=0,
            metadata=metadata,
        )
        out = capsys.readouterr().out
        assert "sessA_spk0" in out and "gender_source=metadata" in out
        # sessB speakers have no metadata entry -> heuristic fallback.
        assert "gender_source=pitch_heuristic" in out

        out_dir2 = tmp_path / "without_meta"
        build(
            bagpiper_corpus["root"], out_dir2, tiny_builder_cfg, seed=0, metadata=None
        )
        out2 = capsys.readouterr().out
        assert "gender_source=metadata" not in out2
        assert "gender_source=pitch_heuristic" in out2


class TestSelectMeasurementTurns:
    """Direct unit tests of the private turn-selection helper: which pool a
    speaker's measurement turns come from (train vs fallback), and the
    earliest-first cumulative cap_sec truncation within that pool."""

    def _turn(self, start, end, speaker="spk0", channel=0, text="hi"):
        return Turn(channel=channel, speaker=speaker, text=text, start=start, end=end)

    def test_speaker_with_train_turns_uses_only_train(self):
        train_turn = self._turn(0.0, 1.0)
        other_turn = self._turn(0.0, 1.0)
        turns_by_speaker = {
            "spk0": [("sessB", other_turn), ("sessA", train_turn)],
        }
        split_by_session = {"sessA": "train", "sessB": "valid"}

        selection = _select_measurement_turns(
            turns_by_speaker, split_by_session, cap_sec=120.0
        )

        picked, source = selection["spk0"]
        assert source == "train"
        assert picked == [("sessA", train_turn)]

    def test_speaker_with_no_train_turns_falls_back_to_any_split(self):
        valid_turn = self._turn(0.0, 1.0)
        test_turn = self._turn(0.0, 1.0)
        turns_by_speaker = {
            "spk1": [("sessC", test_turn), ("sessB", valid_turn)],
        }
        split_by_session = {"sessB": "valid", "sessC": "test"}

        selection = _select_measurement_turns(
            turns_by_speaker, split_by_session, cap_sec=120.0
        )

        picked, source = selection["spk1"]
        assert source == "fallback_any_split"
        # Earliest-first by (session_id, start): sessB before sessC.
        assert [sid for sid, _ in picked] == ["sessB", "sessC"]

    def test_cap_boundary_truncates_after_the_turn_that_reaches_the_cap(self):
        """Two 60s turns land exactly at the 120s cap_sec and are both kept;
        a third 60s turn pushes past the cap and is dropped. The cap check
        runs before appending the *next* candidate, not before appending the
        one that reaches it, so the turn that lands on the boundary is kept."""
        t1 = self._turn(0.0, 60.0)
        t2 = self._turn(60.0, 120.0)
        t3 = self._turn(120.0, 180.0)
        turns_by_speaker = {"spk0": [("sessA", t1), ("sessA", t2), ("sessA", t3)]}
        split_by_session = {"sessA": "train"}

        selection = _select_measurement_turns(
            turns_by_speaker, split_by_session, cap_sec=120.0
        )

        picked, source = selection["spk0"]
        assert source == "train"
        assert [t.start for _, t in picked] == [0.0, 60.0]

    def test_cap_boundary_a_single_oversized_turn_is_still_picked(self):
        """A speaker whose very first turn alone exceeds cap_sec still gets
        that one turn (the picked-so-far guard prevents an empty selection),
        but nothing after it."""
        oversized = self._turn(0.0, 200.0)
        following = self._turn(200.0, 210.0)
        turns_by_speaker = {"spk0": [("sessA", oversized), ("sessA", following)]}
        split_by_session = {"sessA": "train"}

        selection = _select_measurement_turns(
            turns_by_speaker, split_by_session, cap_sec=120.0
        )

        picked, source = selection["spk0"]
        assert picked == [("sessA", oversized)]
        assert source == "train"

    def test_speaker_with_no_turns_anywhere_raises_value_error(self):
        """Contract: a speaker key with an empty turn list is a caller bug
        (turns_by_speaker is normally only populated from observed turns),
        so this raises loudly rather than silently selecting nothing and
        letting a confusing failure surface later inside measure_speaker."""
        turns_by_speaker = {"spk0": []}
        split_by_session: dict[str, str] = {}

        with pytest.raises(ValueError, match="spk0"):
            _select_measurement_turns(turns_by_speaker, split_by_session, cap_sec=120.0)


class TestBuildStatsJson:
    """build() must also persist a machine-readable summary alongside the
    human stdout block, for downstream tooling (e.g. the Delta training
    launch) that shouldn't have to scrape printed text."""

    def test_build_stats_json_matches_stdout_table(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path, capsys
    ):
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)
        stdout = capsys.readouterr().out

        stats_path = out_dir / "build_stats.json"
        assert stats_path.is_file()
        stats = json.loads(stats_path.read_text())

        assert stats["seed"] == 0
        assert set(stats.keys()) >= {
            "seed",
            "root",
            "sessions",
            "splits",
            "speakers",
            "caption_word_length",
        }

        for split in SPLITS:
            entry = stats["splits"][split]
            assert set(entry.keys()) >= {
                "sessions",
                "windows",
                "tac_dropped",
                "tac_records",
                "mono_records",
            }

        train_mono, _ = _load_variant_split(out_dir, "mono", "train")
        train_tac, _ = _load_variant_split(out_dir, "tac", "train")
        assert stats["splits"]["train"]["mono_records"] == len(train_mono)
        assert stats["splits"]["train"]["tac_records"] == len(train_tac)
        assert stats["splits"]["valid"]["windows"] == 0
        assert stats["splits"]["test"]["windows"] == 0

        speaker_ids = {s["speaker_id"] for s in stats["speakers"]}
        assert speaker_ids == {"sessA_spk0", "sessA_spk1", "sessB_spk0", "sessB_spk1"}
        for entry in stats["speakers"]:
            assert set(entry.keys()) >= {
                "speaker_id",
                "pitch_band",
                "variability_band",
                "rate_band",
                "gender",
                "gender_source",
                "measure_source",
                "voice_description",
            }
            # Cross-check against the printed stats table (same build run).
            assert f"{entry['speaker_id']}:" in stdout
            assert f"measure_source={entry['measure_source']}" in stdout
            assert entry["voice_description"] in stdout

        cwl = stats["caption_word_length"]
        assert "_note" in cwl and "token" in cwl["_note"]
        for variant in VARIANTS:
            for split in SPLITS:
                assert split in cwl[variant]
        # Fixture's 2 sessions both land in train -> valid/test are empty.
        assert cwl["tac"]["valid"] is None
        assert cwl["mono"]["test"] is None

        train_dist = cwl["mono"]["train"]
        assert set(train_dist.keys()) == {
            "n",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "mean",
        }
        assert train_dist["n"] == len(train_mono)

    def test_human_stdout_stats_block_unchanged(
        self, bagpiper_corpus, tiny_builder_cfg, tmp_path, capsys
    ):
        """Fix 2 must not alter the existing human-readable stdout block."""
        out_dir = tmp_path / "out"
        build(bagpiper_corpus["root"], out_dir, tiny_builder_cfg, seed=0)
        out = capsys.readouterr().out
        assert "BagPiper lm_tts dataset build" in out
        assert "windows" in out
        assert "tac-dropped" in out
        assert "gender_source" in out
        assert "caption length" in out


class TestSplitSessions:
    def test_deterministic_and_ratios(self):
        ids = [f"s{i:03d}" for i in range(100)]
        ratios = {"train": 0.96, "valid": 0.02, "test": 0.02}
        a = split_sessions(ids, ratios, seed=0)
        b = split_sessions(ids, ratios, seed=0)
        assert a == b
        assert len(a["train"]) == 96 and len(a["valid"]) == 2 and len(a["test"]) == 2
        assert sorted(a["train"] + a["valid"] + a["test"]) == ids
        assert split_sessions(ids, ratios, seed=1) != a

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            split_sessions(["a", "b"], {"train": 0.5, "valid": 0.1, "test": 0.1}, 0)


class TestResolveDatasetRoot:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("SSSD_ROOT", "/env/root")
        assert resolve_dataset_root(
            "/explicit/root", {"dataset_root": "/cfg/root"}
        ) == Path("/explicit/root")

    def test_env_var_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("SSSD_ROOT", "/env/root")
        assert resolve_dataset_root(None, {"dataset_root": "/cfg/root"}) == Path(
            "/env/root"
        )

    def test_config_fallback(self, monkeypatch):
        monkeypatch.delenv("SSSD_ROOT", raising=False)
        assert resolve_dataset_root(None, {"dataset_root": "/cfg/root"}) == Path(
            "/cfg/root"
        )


class TestRealConfigCopiesF5Values:
    """conf/dataset.yaml must carry the same windowing/split parameter
    VALUES as the F5 recipe's config.yaml (brief requirement)."""

    def test_values_match_f5_reference(self):
        conf_path = Path(__file__).resolve().parents[2] / "conf" / "dataset.yaml"
        cfg = yaml.safe_load(conf_path.read_text())["builder"]
        assert cfg["window_min"] == 10.0
        assert cfg["window_max"] == 60.0
        assert cfg["boundary_guard"] == 0.0
        assert cfg["tail_min"] == 5.0
        assert cfg["merge_gap"] == 1.0
        assert cfg["seed"] == 0
        assert cfg["split_ratios"] == {"train": 0.96, "valid": 0.02, "test": 0.02}
        assert cfg["source_sample_rate"] == 48000

    def test_load_config_reads_builder_block(self):
        conf_path = Path(__file__).resolve().parents[2] / "conf" / "dataset.yaml"
        cfg = load_config(conf_path)
        assert cfg["window_min"] == 10.0


class TestCLI:
    def test_main_builds_via_argv(self, bagpiper_corpus, tmp_path, monkeypatch, capsys):
        conf_path = tmp_path / "tiny.yaml"
        conf_path.write_text(
            yaml.safe_dump(
                {
                    "builder": {
                        "manifests_subdir": "lhotse_manifests_48",
                        "audio_subdir": "original",
                        "merge_gap": 0.5,
                        "window_min": 3.0,
                        "window_max": 6.0,
                        "boundary_guard": 0.0,
                        "tail_min": 1.0,
                        "split_ratios": {"train": 0.96, "valid": 0.02, "test": 0.02},
                        "source_sample_rate": 48000,
                        "target_sample_rate": 16000,
                        "measure_cap_sec": 120.0,
                        "seed": 0,
                        "dataset_root": str(bagpiper_corpus["root"]),
                    }
                }
            )
        )
        out_dir = tmp_path / "cli_out"
        argv = [
            "--dataset-root",
            str(bagpiper_corpus["root"]),
            "--out-dir",
            str(out_dir),
            "--conf",
            str(conf_path),
        ]
        monkeypatch.setattr("sys.argv", ["builder"] + argv)
        main()
        assert (out_dir / "mono" / "train" / "dialogues.jsonl").is_file()
        capsys.readouterr()
