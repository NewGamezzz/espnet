"""End-to-end builder tests over the fabricated mini corpus (AC5, AC7, AC9)."""

import json

import pytest

from egs3.conversational.tts.dataset.builder import (
    SSSDBuilder,
    overlap_and_speech_time,
    split_sessions,
)
from egs3.conversational.tts.dataset.dataset import (
    ConversationDataset,
    collate_conversations,
)
from egs3.conversational.tts.dataset.sssd import Turn
from egs3.conversational.tts.dataset.text import NEW_TOKENS


def build(fake_corpus, recipe_dir=None, seed=0):
    recipe_dir = recipe_dir or fake_corpus["recipe_dir"]
    builder = SSSDBuilder()
    builder.build(
        recipe_dir=recipe_dir,
        dataset_root=fake_corpus["root"],
        seed=seed,
        base_vocab_path=fake_corpus["base_vocab_path"],
    )
    return builder, recipe_dir


class TestBuildEndToEnd:
    def test_source_checks(self, fake_corpus, tmp_path):
        builder = SSSDBuilder()
        assert builder.is_source_prepared(dataset_root=fake_corpus["root"])
        empty = tmp_path / "nothing"
        assert not builder.is_source_prepared(dataset_root=empty)
        with pytest.raises(RuntimeError, match="externally"):
            builder.prepare_source(dataset_root=empty)

    def test_outputs_exist_and_windows_are_sane(self, fake_corpus, capsys):
        builder, recipe_dir = build(fake_corpus)
        assert builder.is_built(recipe_dir=recipe_dir)
        data_dir = recipe_dir / "data"
        train = (data_dir / "manifest/train.jsonl").read_text().splitlines()
        assert train, "3 sessions with ratios 0.96/0.02/0.02 must all land in train"
        sessions = set()
        for line in train:
            w = json.loads(line)
            assert w["audio_relpath"].startswith("original/")
            assert 0 <= w["t0"] < w["t1"]
            assert w["turns"], "zero-speech windows must be dropped"
            for turn in w["turns"]:
                assert w["t0"] <= turn["start"] and turn["end"] <= w["t1"]
                assert turn["text"] == turn["text"].lower()
            sessions.add(w["session_id"])
        assert sessions == {"sess_long", "sess_tri", "sess_short"}
        summary = capsys.readouterr().out
        assert "speaker overlap" in summary
        assert "overlap ratio" in summary

    def test_full_path_to_collated_batch(self, fake_corpus):
        """AC7: turns -> windows -> dataset item -> packed batch, incl. N=3."""
        _, recipe_dir = build(fake_corpus)
        ds = ConversationDataset(
            split="train", recipe_dir=recipe_dir, dataset_root=fake_corpus["root"]
        )
        batch = collate_conversations([ds[i] for i in range(len(ds))])
        assert set(batch["counts"]) == {2, 3}
        assert batch["speech"].shape[0] == sum(batch["counts"])
        assert batch["text"].shape[0] == sum(batch["counts"])
        assert (batch["text"] >= -1).all()

    def test_determinism_byte_identical(self, fake_corpus, tmp_path):
        """AC5: two builds with the same seed produce identical artifacts."""
        _, dir_a = build(fake_corpus, recipe_dir=tmp_path / "a")
        _, dir_b = build(fake_corpus, recipe_dir=tmp_path / "b")
        for relpath in (
            "manifest/train.jsonl",
            "manifest/valid.jsonl",
            "manifest/test.jsonl",
            "tokens/vocab.txt",
        ):
            assert (dir_a / "data" / relpath).read_bytes() == (
                dir_b / "data" / relpath
            ).read_bytes(), relpath


class TestVocabSafety:
    """AC9: base ids unchanged, new ids contiguous at the end."""

    def test_vocab_file_and_meta(self, fake_corpus, base_vocab):
        _, recipe_dir = build(fake_corpus)
        data_dir = recipe_dir / "data"
        base_bytes = fake_corpus["base_vocab_path"].read_bytes()
        vocab_bytes = (data_dir / "tokens/vocab.txt").read_bytes()
        assert vocab_bytes == base_bytes + "\n".join(NEW_TOKENS).encode() + b"\n"
        meta = json.loads((data_dir / "tokens/vocab_meta.json").read_text())
        assert meta["base_size"] == len(base_vocab)
        assert meta["total_size"] == len(base_vocab) + 2
        assert meta["new_tokens"] == {
            NEW_TOKENS[0]: len(base_vocab),
            NEW_TOKENS[1]: len(base_vocab) + 1,
        }

    def test_missing_base_vocab_fails_loudly(self, fake_corpus, tmp_path):
        builder = SSSDBuilder()
        with pytest.raises(ValueError, match="base_vocab_path is required"):
            builder.build(
                recipe_dir=fake_corpus["recipe_dir"],
                dataset_root=fake_corpus["root"],
                seed=0,
            )
        with pytest.raises(FileNotFoundError):
            builder.build(
                recipe_dir=fake_corpus["recipe_dir"],
                dataset_root=fake_corpus["root"],
                seed=0,
                base_vocab_path=tmp_path / "missing.txt",
            )


class TestSplitAndStats:
    def test_split_sessions_ratios_and_determinism(self):
        ids = [f"s{i:03d}" for i in range(100)]
        ratios = {"train": 0.96, "valid": 0.02, "test": 0.02}
        a = split_sessions(ids, ratios, seed=0)
        b = split_sessions(ids, ratios, seed=0)
        assert a == b
        assert len(a["test"]) == 2 and len(a["valid"]) == 2 and len(a["train"]) == 96
        assert sorted(a["train"] + a["valid"] + a["test"]) == ids
        assert split_sessions(ids, ratios, seed=1) != a

    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            split_sessions(["a", "b"], {"train": 0.5, "valid": 0.1, "test": 0.1}, 0)

    def test_overlap_and_speech_time(self):
        turns = [
            Turn(0, "a", "x", 0.0, 2.0),
            Turn(1, "b", "y", 1.0, 3.0),  # 1 s pairwise overlap
            Turn(0, "a", "z", 10.0, 11.0),
        ]
        overlap, speech = overlap_and_speech_time(turns)
        assert overlap == pytest.approx(1.0)
        assert speech == pytest.approx(4.0)


class TestCLI:
    def test_main_builds_and_skips_when_built(self, fake_corpus, monkeypatch, capsys):
        from egs3.conversational.tts.dataset import builder as builder_mod

        argv = [
            "builder",
            "--recipe-dir",
            str(fake_corpus["recipe_dir"]),
            "--dataset-root",
            str(fake_corpus["root"]),
            "--base-vocab-path",
            str(fake_corpus["base_vocab_path"]),
        ]
        monkeypatch.setattr("sys.argv", argv)
        builder_mod.main()
        assert SSSDBuilder().is_built(recipe_dir=fake_corpus["recipe_dir"])
        capsys.readouterr()
        builder_mod.main()  # second run: already built, no --force
        assert "Already built" in capsys.readouterr().out
