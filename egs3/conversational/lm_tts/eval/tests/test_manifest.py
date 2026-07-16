"""Tests for the eval manifest builders (Task 2): Set A (SSSD mono) and
Set B (native SFT dev_multi_talker), converted into the one manifest schema
documented in the "BagPiper Conversational Baseline Eval" plan:

    {"example_id": ..., "set": "sssd"|"sft", "system": ..., "caption": ...,
     "gt_wav": ..., "turns": [...], "speakers": [...]|None,
     "ref_wavs": {...}|None}
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.manifest import (
    build_manifest_sft,
    build_manifest_sssd,
    load_manifest,
    parse_sft_turns,
    write_manifest,
)

SYSTEM_PROMPT = "You are a multi-talker text-to-speech system."


# ---------------------------------------------------------------------------
# Set A: SSSD mono fixtures (Task 1 metadata shape)
# ---------------------------------------------------------------------------


def _sssd_record(window_id: str, speakers: list[str]) -> dict:
    mix_path = f"/abs/sssd/{window_id}_mix.wav"
    channel_wavs = {
        spk: f"/abs/sssd/{window_id}_ch{i}.wav" for i, spk in enumerate(speakers)
    }
    turns = [
        {
            "speaker": spk,
            "start": round(0.5 + i * 1.5, 3),
            "end": round(1.5 + i * 1.5, 3),
            "text": f"{spk} says hello number {i}",
        }
        for i, spk in enumerate(speakers)
    ]
    return {
        "example_id": f"sssd_mono_{window_id}",
        "messages": [
            ["system", "text", SYSTEM_PROMPT],
            ["user", "text", f"Two speakers narrate a conversation for {window_id}."],
            ["assistant", "text", "<think>plan</think>"],
            ["assistant", "audio", mix_path],
        ],
        "metadata": {
            "conv_id": window_id,
            "variant": "mono",
            "speakers": speakers,
            "t0": 0.0,
            "t1": 10.0,
            "turns": turns,
            "channel_wavs": channel_wavs,
        },
    }


@pytest.fixture()
def sssd_dialogues_path(tmp_path: Path) -> Path:
    records = [
        _sssd_record("sess1_w00000", ["spkA", "spkB"]),
        _sssd_record("sess1_w00001", ["spkC", "spkD"]),
    ]
    path = tmp_path / "sssd_dialogues.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


class TestBuildManifestSssd:
    def test_entry_fields_exact(self, sssd_dialogues_path: Path):
        entries = build_manifest_sssd(sssd_dialogues_path)
        assert len(entries) == 2
        entry = next(e for e in entries if e["example_id"] == "sssd_mono_sess1_w00000")
        assert entry["set"] == "sssd"
        assert entry["system"] == SYSTEM_PROMPT
        assert entry["caption"] == "Two speakers narrate a conversation for sess1_w00000."
        assert entry["gt_wav"] == "/abs/sssd/sess1_w00000_mix.wav"
        assert entry["speakers"] == ["spkA", "spkB"]

    def test_turns_passed_through(self, sssd_dialogues_path: Path):
        entries = build_manifest_sssd(sssd_dialogues_path)
        entry = next(e for e in entries if e["example_id"] == "sssd_mono_sess1_w00000")
        expected_turns = [
            {"speaker": "spkA", "start": 0.5, "end": 1.5, "text": "spkA says hello number 0"},
            {"speaker": "spkB", "start": 2.0, "end": 3.0, "text": "spkB says hello number 1"},
        ]
        assert entry["turns"] == expected_turns

    def test_ref_wavs_equals_channel_wavs(self, sssd_dialogues_path: Path):
        entries = build_manifest_sssd(sssd_dialogues_path)
        entry = next(e for e in entries if e["example_id"] == "sssd_mono_sess1_w00000")
        assert entry["ref_wavs"] == {
            "spkA": "/abs/sssd/sess1_w00000_ch0.wav",
            "spkB": "/abs/sssd/sess1_w00000_ch1.wav",
        }

    def test_limit_and_seed_are_deterministic(self, sssd_dialogues_path: Path):
        first = build_manifest_sssd(sssd_dialogues_path, limit=1, seed=7)
        second = build_manifest_sssd(sssd_dialogues_path, limit=1, seed=7)
        assert first == second
        assert len(first) == 1

    def test_limit_larger_than_available_takes_all(self, sssd_dialogues_path: Path):
        entries = build_manifest_sssd(sssd_dialogues_path, limit=100, seed=7)
        assert len(entries) == 2

    def test_entries_sorted_by_example_id(self, sssd_dialogues_path: Path):
        entries = build_manifest_sssd(sssd_dialogues_path)
        ids = [e["example_id"] for e in entries]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Set B: native SFT dev_multi_talker fixtures (real record shape)
# ---------------------------------------------------------------------------


def _sft_record(example_id: str, utt_id: str, caption: str, wav_relpath: str) -> dict:
    return {
        "example_id": example_id,
        "metadata": {"utt_id": utt_id},
        "messages": [
            ["system", "text", SYSTEM_PROMPT],
            ["user", "text", caption],
            ["assistant", "text", "<think>plan</think>"],
            ["assistant", "audio", f"/mnt/sft-source/{wav_relpath}"],
        ],
    }


CURLY_CAPTION = (
    "A British female narrator speaks with a male companion. "
    "She says: “Later that week I dropped by the office.” "
    "He replies: “That sounds delightful.”"
)
STRAIGHT_CAPTION = 'A calm male narrator speaks. He says: "Good evening everyone."'
ZERO_QUOTE_CAPTION = "A narrator speaks without any quoted dialogue at all."


@pytest.fixture()
def sft_dialogues_path(tmp_path: Path) -> Path:
    records = [
        _sft_record(
            "multi_talker_tts_YOU1000000035_M0000024",
            "YOU1000000035_M0000024",
            CURLY_CAPTION,
            "YOU1000000035/YOU1000000035_M0000024.wav",
        ),
        _sft_record(
            "multi_talker_tts_YOU2000000099_M0000001",
            "YOU2000000099_M0000001",
            STRAIGHT_CAPTION,
            "YOU2000000099/YOU2000000099_M0000001.wav",
        ),
    ]
    path = tmp_path / "sft_dialogues.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture()
def sft_audio_root(tmp_path: Path) -> Path:
    root = tmp_path / "audio_root"
    for show, utt_id in [
        ("YOU1000000035", "YOU1000000035_M0000024"),
        ("YOU2000000099", "YOU2000000099_M0000001"),
    ]:
        show_dir = root / show
        show_dir.mkdir(parents=True, exist_ok=True)
        (show_dir / f"{utt_id}.wav").write_bytes(b"RIFF....WAVEfmt ")
    return root


class TestParseSftTurns:
    def test_curly_quotes_in_order(self):
        turns = parse_sft_turns(CURLY_CAPTION)
        assert turns == [
            "Later that week I dropped by the office.",
            "That sounds delightful.",
        ]

    def test_straight_quotes_fallback(self):
        turns = parse_sft_turns(STRAIGHT_CAPTION)
        assert turns == ["Good evening everyone."]

    def test_no_quotes_returns_empty(self):
        assert parse_sft_turns(ZERO_QUOTE_CAPTION) == []

    def test_never_mixes_curly_and_straight(self):
        mixed = 'She says: “Curly one.” and also "a straight one".'
        assert parse_sft_turns(mixed) == ["Curly one."]


class TestBuildManifestSft:
    def test_entry_fields_and_null_speaker_fields(
        self, sft_dialogues_path: Path, sft_audio_root: Path
    ):
        entries = build_manifest_sft(sft_dialogues_path, sft_audio_root)
        entry = next(
            e
            for e in entries
            if e["example_id"] == "multi_talker_tts_YOU1000000035_M0000024"
        )
        assert entry["set"] == "sft"
        assert entry["system"] == SYSTEM_PROMPT
        assert entry["caption"] == CURLY_CAPTION
        assert entry["gt_wav"] == str(
            sft_audio_root / "YOU1000000035" / "YOU1000000035_M0000024.wav"
        )
        assert entry["speakers"] is None
        assert entry["ref_wavs"] is None
        assert entry["turns"] == [
            {
                "speaker": None,
                "start": None,
                "end": None,
                "text": "Later that week I dropped by the office.",
            },
            {"speaker": None, "start": None, "end": None, "text": "That sounds delightful."},
        ]

    def test_straight_quote_record_turns(
        self, sft_dialogues_path: Path, sft_audio_root: Path
    ):
        entries = build_manifest_sft(sft_dialogues_path, sft_audio_root)
        entry = next(
            e
            for e in entries
            if e["example_id"] == "multi_talker_tts_YOU2000000099_M0000001"
        )
        assert [t["text"] for t in entry["turns"]] == ["Good evening everyone."]

    def test_zero_quote_caption_raises_value_error_naming_id(
        self, tmp_path: Path, sft_audio_root: Path
    ):
        bad_id = "multi_talker_tts_YOU3000000000_M0000002"
        record = _sft_record(
            bad_id,
            "YOU3000000000_M0000002",
            ZERO_QUOTE_CAPTION,
            "YOU3000000000/YOU3000000000_M0000002.wav",
        )
        path = tmp_path / "bad_dialogues.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match=bad_id):
            build_manifest_sft(path, sft_audio_root)

    def test_missing_gt_wav_raises_file_not_found_with_path(
        self, sft_dialogues_path: Path, tmp_path: Path
    ):
        empty_root = tmp_path / "empty_audio_root"
        empty_root.mkdir()
        expected_path = empty_root / "YOU1000000035" / "YOU1000000035_M0000024.wav"
        with pytest.raises(FileNotFoundError, match=str(expected_path)):
            build_manifest_sft(sft_dialogues_path, empty_root)


# ---------------------------------------------------------------------------
# write_manifest / load_manifest roundtrip
# ---------------------------------------------------------------------------


class TestManifestRoundtrip:
    def test_write_then_load_is_identity(
        self, sssd_dialogues_path: Path, tmp_path: Path
    ):
        entries = build_manifest_sssd(sssd_dialogues_path)
        out_path = tmp_path / "manifest.json"
        write_manifest(entries, out_path)
        loaded = load_manifest(out_path)
        assert loaded == entries

    def test_written_file_is_json_list_indent_1(
        self, sssd_dialogues_path: Path, tmp_path: Path
    ):
        entries = build_manifest_sssd(sssd_dialogues_path)
        out_path = tmp_path / "manifest.json"
        write_manifest(entries, out_path)
        raw = out_path.read_text(encoding="utf-8")
        assert json.loads(raw) == entries
        assert isinstance(json.loads(raw), list)
        # indent=1 means nested keys are indented 1 space per nesting level;
        # entries are list items (level 1) containing dict keys (level 2).
        assert raw.startswith("[\n")
        assert '\n  "example_id"' in raw
