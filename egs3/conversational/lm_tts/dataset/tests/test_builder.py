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
    build,
    load_config,
    main,
    resolve_dataset_root,
    split_sessions,
)

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
