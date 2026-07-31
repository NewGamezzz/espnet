"""LibriTTS scanning and utterance-as-window record construction."""

from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.preprocessing.libritts import (
    UttEntry,
    scan_subset,
    subsample_to_hours,
    utterance_record,
)

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)


def make_tree(root: Path, utts: dict[str, str]) -> None:
    """utts: relpath-without-extension -> transcript. Writes .normalized.txt
    and a tiny placeholder .wav file (content irrelevant for scanning)."""
    for rel, text in utts.items():
        base = root / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_name(base.name + ".normalized.txt").write_text(
            text, encoding="utf-8"
        )
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


def test_utterance_record_shape(tmp_path):
    entry = UttEntry(
        utt_id="103_1241_000000_000001",
        audio_relpath="train-clean-100/103/1241/103_1241_000000_000001.wav",
        speaker="103",
        chapter="1241",
        text="Hello there.",
    )
    record = utterance_record(entry, duration=2.5, sample_rate=24000, text="hello there.")
    assert record.window_id == "libritts_103_1241_000000_000001"
    assert record.session_id == "libritts_103_1241"
    assert record.num_channels == 1
    assert record.sample_rate == 24000
    assert (record.t0, record.t1) == (0.0, 2.5)
    assert record.num_active_speakers == 1
    assert record.exchange_count == 0
    (turn,) = record.turns
    assert (turn.channel, turn.speaker) == (0, "103")
    assert turn.text == "hello there."  # normalized text, not the raw transcript
    assert (turn.start, turn.end) == (0.0, 2.5)


def test_subsample_to_hours_budget_and_determinism():
    items = [
        (
            UttEntry(
                utt_id=f"u{i}", audio_relpath=f"u{i}.wav",
                speaker="s", chapter="c", text="t",
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
