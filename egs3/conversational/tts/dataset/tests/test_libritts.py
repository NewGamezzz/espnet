"""LibriTTS scanning and utterance-as-session/window record construction."""

import string
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.dataset.libritts_builder import LibriTTSBuilder
from egs3.conversational.tts.dataset.preprocessing.libritts import (
    UttEntry,
    scan_subset,
    subsample_to_hours,
    utterance_session,
)
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    read_session_manifest,
)

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)


def make_tree(root: Path, utts: dict[str, str]) -> None:
    """utts: relpath-without-extension -> transcript. Writes .normalized.txt
    and a tiny placeholder .wav file (content irrelevant for scanning)."""
    for rel, text in utts.items():
        base = root / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_name(base.name + ".normalized.txt").write_text(text, encoding="utf-8")
        base.with_name(base.name + ".wav").write_bytes(b"\x00")


def test_scan_subset_pairs_and_ids(tmp_path):
    make_tree(
        tmp_path,
        {
            "train-clean-100/103/1241/103_1241_000000_000001": "Hello there.",
            "train-clean-100/103/1241/103_1241_000000_000002": "Second one.",
            "train-clean-100/911/128684/911_128684_000004_000000": "Other speaker.",
        },
    )
    entries = scan_subset(tmp_path, "train-clean-100")
    assert [e.utt_id for e in entries] == [
        "103_1241_000000_000001",
        "103_1241_000000_000002",
        "911_128684_000004_000000",
    ]
    first = entries[0]
    assert first.speaker == "103"
    assert first.chapter == "1241"
    assert first.text == "Hello there."
    assert first.audio_relpath == (
        "train-clean-100/103/1241/103_1241_000000_000001.wav"
    )


def test_scan_subset_skips_missing_wav_and_empty_text(tmp_path):
    make_tree(tmp_path, {"train-clean-100/1/2/1_2_000000_000001": "kept"})
    # transcript without wav
    orphan = tmp_path / "train-clean-100/1/2/1_2_000000_000002.normalized.txt"
    orphan.write_text("no wav", encoding="utf-8")
    # empty transcript with wav
    make_tree(tmp_path, {"train-clean-100/1/2/1_2_000000_000003": "   "})
    entries = scan_subset(tmp_path, "train-clean-100")
    assert [e.utt_id for e in entries] == ["1_2_000000_000001"]


def test_scan_subset_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_subset(tmp_path, "train-clean-360")


def test_scan_subset_threaded_matches_serial(tmp_path):
    make_tree(
        tmp_path,
        {
            f"train-clean-100/{spk}/{ch}/{spk}_{ch}_000000_00000{i}": f"utt {i}"
            for spk, ch in (("103", "1241"), ("911", "128684"))
            for i in range(5)
        },
    )
    # orphan transcript exercises the skip path under threading too
    orphan = tmp_path / "train-clean-100/103/1241/103_1241_000000_000009.normalized.txt"
    orphan.write_text("no wav", encoding="utf-8")
    serial = scan_subset(tmp_path, "train-clean-100")
    threaded = scan_subset(tmp_path, "train-clean-100", workers=4)
    assert threaded == serial
    assert len(threaded) == 10


def test_utterance_session_shape(tmp_path):
    entry = UttEntry(
        utt_id="103_1241_000000_000001",
        audio_relpath="train-clean-100/103/1241/103_1241_000000_000001.wav",
        speaker="103",
        chapter="1241",
        text="Hello there.",
    )
    record = utterance_session(
        entry, duration=2.5, sample_rate=24000, text="hello there."
    )
    assert record.session_id == "libritts_103_1241"
    assert record.audio_relpath == entry.audio_relpath
    assert record.num_channels == 1
    assert record.sample_rate == 24000
    assert record.duration == 2.5
    assert record.atomic is True
    assert record.window_id == "libritts_103_1241_000000_000001"
    (turn,) = record.turns
    assert (turn.channel, turn.speaker) == (0, "103")
    assert turn.text == "hello there."  # normalized text, not the raw transcript
    assert (turn.start, turn.end) == (0.0, 2.5)  # turn spans the whole utterance


def test_subsample_to_hours_budget_and_determinism():
    items = [
        (
            UttEntry(
                utt_id=f"u{i}",
                audio_relpath=f"u{i}.wav",
                speaker="s",
                chapter="c",
                text="t",
            ),
            60.0,  # one minute each
        )
        for i in range(100)
    ]
    taken = subsample_to_hours(items, hours=0.5, seed=0)
    total = sum(dur for _, dur in taken)
    assert 0.5 * 3600 <= total < 0.5 * 3600 + 60.0  # stops right after the budget
    assert taken == subsample_to_hours(items, hours=0.5, seed=0)  # deterministic
    other = subsample_to_hours(items, hours=0.5, seed=1)
    assert {e.utt_id for e, _ in taken} != {e.utt_id for e, _ in other}
    # output is sorted by utt_id for stable manifest order
    assert [e.utt_id for e, _ in taken] == sorted(e.utt_id for e, _ in taken)


def write_wav(path: Path, duration_s: float, sr: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(sr * duration_s)) / sr
    sf.write(str(path), 0.1 * np.sin(2 * np.pi * 440.0 * t), sr)


def fabricate_corpus(root: Path) -> None:
    """Minimal LibriTTS tree: all builder subsets present."""
    utts = {
        # train subsets: one keeper each, one too-short in clean-100
        "train-clean-100/1/10/1_10_000000_000001": ("Hello there.", 2.0),
        "train-clean-100/1/10/1_10_000000_000002": ("Too short.", 0.5),
        "train-clean-360/2/20/2_20_000000_000001": ("Second subset.", 1.5),
        "train-other-500/3/30/3_30_000000_000001": ("Third subset.", 1.2),
        "dev-clean/4/40/4_40_000000_000001": ("Valid one.", 2.0),
        "dev-clean/4/40/4_40_000000_000002": ("Valid two.", 1.1),
    }
    for rel, (text, dur) in utts.items():
        base = root / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_name(base.name + ".normalized.txt").write_text(text, encoding="utf-8")
        write_wav(base.with_name(base.name + ".wav"), dur)


def fabricate_recipe(recipe_dir: Path) -> None:
    """A recipe dir with a prebuilt extended vocab (SSSD build stand-in)."""
    tokens = [" "] + list(string.ascii_lowercase) + [".", ","] + ["<turn>", "<OTHER>"]
    vocab = recipe_dir / "data/tokens/vocab.txt"
    vocab.parent.mkdir(parents=True, exist_ok=True)
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def test_builder_end_to_end(tmp_path):
    root, recipe = tmp_path / "LibriTTS", tmp_path / "recipe"
    fabricate_corpus(root)
    fabricate_recipe(recipe)
    builder = LibriTTSBuilder()
    assert builder.is_source_prepared(dataset_root=root)
    assert not builder.is_built(recipe_dir=recipe)
    builder.build(recipe_dir=recipe, dataset_root=root, seed=0)
    assert builder.is_built(recipe_dir=recipe)

    train = read_session_manifest(
        recipe / "data/manifest/sessions_libritts_train.jsonl"
    )
    # the 0.5 s utterance is dropped; the three subsets each contribute one
    assert sorted(r.window_id for r in train) == [
        "libritts_1_10_000000_000001",
        "libritts_2_20_000000_000001",
        "libritts_3_30_000000_000001",
    ]
    for r in train:
        assert r.atomic is True
        assert r.num_channels == 1
        assert r.duration >= 1.0
        assert r.turns[0].channel == 0
        # normalized against the charset: lowercased, punctuation kept
        assert r.turns[0].text == r.turns[0].text.lower()

    valid = read_session_manifest(
        recipe / "data/manifest/sessions_libritts_valid.jsonl"
    )
    assert {r.window_id for r in valid} == {
        "libritts_4_40_000000_000001",
        "libritts_4_40_000000_000002",
    }


def test_builder_accepts_libritts_root_passthrough(tmp_path):
    """The create_dataset stage passes the training config's libritts_root
    (shared kwargs; other builders ignore it via **_) so the build root can
    never drift from the dataloader's audio root."""
    root, recipe = tmp_path / "LibriTTS", tmp_path / "recipe"
    fabricate_corpus(root)
    fabricate_recipe(recipe)
    builder = LibriTTSBuilder()
    assert builder.is_source_prepared(libritts_root=root)
    builder.prepare_source(recipe_dir=recipe, libritts_root=root)
    builder.build(recipe_dir=recipe, libritts_root=root, seed=0)
    assert builder.is_built(recipe_dir=recipe)
    train = read_session_manifest(
        recipe / "data/manifest/sessions_libritts_train.jsonl"
    )
    assert len(train) == 3


def test_builder_requires_vocab(tmp_path):
    root, recipe = tmp_path / "LibriTTS", tmp_path / "recipe"
    fabricate_corpus(root)
    with pytest.raises(RuntimeError, match="SSSD build"):
        LibriTTSBuilder().build(recipe_dir=recipe, dataset_root=root)


def test_builder_rejects_sample_rate_mismatch(tmp_path):
    root, recipe = tmp_path / "LibriTTS", tmp_path / "recipe"
    fabricate_corpus(root)
    fabricate_recipe(recipe)
    # Overwrite one utterance's wav at 48 kHz; config.yaml's libritts_builder
    # pins sample_rate: 24000, so the builder must fail loudly rather than
    # silently mixing rates into the manifest.
    write_wav(root / "train-clean-100/1/10/1_10_000000_000001.wav", 2.0, sr=48000)
    with pytest.raises(RuntimeError, match="sample rate"):
        LibriTTSBuilder().build(recipe_dir=recipe, dataset_root=root, seed=0)


def test_manifest_loads_through_conversation_dataset(tmp_path):
    root, recipe = tmp_path / "LibriTTS", tmp_path / "recipe"
    fabricate_corpus(root)
    fabricate_recipe(recipe)
    LibriTTSBuilder().build(recipe_dir=recipe, dataset_root=root, seed=0)
    dataset = ConversationDataset(
        split="train",
        manifest_path=recipe / "data/manifest/sessions_libritts_train.jsonl",
        dataset_root=root,
        fs=24000,
    )
    sample = dataset[0]
    assert sample["num_channels"] == 1
    assert sample["speech"].shape[0] == 1
    assert sample["speech"].shape[1] >= 24000  # >= 1 s at 24 kHz, no resample
    assert sample["perm"].tolist() == [0]
    assert sample["turns"][0].channel == 0
