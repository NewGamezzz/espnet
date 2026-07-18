"""Tests for the LibriSpeech test-clean manifest builder."""

from __future__ import annotations

import json

import pytest

from eval.manifest_librispeech import (
    CAPTION_TEMPLATE,
    SYSTEM_PROMPT,
    build_manifest_librispeech,
)


def _make_corpus(root, chapters):
    """chapters: {(spk, chap): [(utt_no, "CAPS TEXT"), ...]}"""
    for (spk, chap), utts in chapters.items():
        d = root / spk / chap
        d.mkdir(parents=True)
        lines = []
        for utt_no, text in utts:
            utt_id = f"{spk}-{chap}-{utt_no}"
            (d / f"{utt_id}.flac").write_bytes(b"\x00")
            lines.append(f"{utt_id} {text}")
        (d / f"{spk}-{chap}.trans.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def test_entry_schema_and_lowercase(tmp_path):
    _make_corpus(tmp_path, {("1089", "134686"): [("0000", "HELLO WORLD")]})

    entries = build_manifest_librispeech(tmp_path)

    assert len(entries) == 1
    e = entries[0]
    assert e["example_id"] == "librispeech_test_clean_1089-134686-0000"
    assert e["set"] == "librispeech"
    assert e["system"] == SYSTEM_PROMPT
    assert e["caption"] == CAPTION_TEMPLATE.format(text="hello world")
    assert e["caption"].endswith('"hello world"')
    assert e["gt_wav"].endswith("1089/134686/1089-134686-0000.flac")
    assert e["turns"] == [
        {"speaker": None, "start": None, "end": None, "text": "hello world"}
    ]
    assert e["speakers"] is None
    assert e["ref_wavs"] is None


def test_walks_all_chapters_sorted(tmp_path):
    _make_corpus(
        tmp_path,
        {
            ("2", "20"): [("0001", "B TEXT")],
            ("1", "10"): [("0001", "A TEXT"), ("0002", "C TEXT")],
        },
    )

    entries = build_manifest_librispeech(tmp_path)

    ids = [e["example_id"] for e in entries]
    assert ids == sorted(ids)
    assert len(entries) == 3


def test_pilot_limit_is_seeded_and_deterministic(tmp_path):
    _make_corpus(
        tmp_path,
        {("1", "10"): [(f"{i:04d}", f"TEXT {i}") for i in range(10)]},
    )

    a = build_manifest_librispeech(tmp_path, limit=4, seed=0)
    b = build_manifest_librispeech(tmp_path, limit=4, seed=0)
    c = build_manifest_librispeech(tmp_path, limit=4, seed=1)

    assert [e["example_id"] for e in a] == [e["example_id"] for e in b]
    assert len(a) == 4
    assert [e["example_id"] for e in a] != [e["example_id"] for e in c]


def test_missing_flac_raises(tmp_path):
    _make_corpus(tmp_path, {("1", "10"): [("0001", "OK TEXT")]})
    (tmp_path / "1" / "10" / "1-10-0001.flac").unlink()

    with pytest.raises(FileNotFoundError, match="1-10-0001.flac"):
        build_manifest_librispeech(tmp_path)


def test_empty_corpus_raises(tmp_path):
    with pytest.raises(ValueError, match="no transcript"):
        build_manifest_librispeech(tmp_path)


def test_transcript_line_without_text_raises(tmp_path):
    d = tmp_path / "1" / "10"
    d.mkdir(parents=True)
    (d / "1-10-0001.flac").write_bytes(b"\x00")
    (d / "1-10.trans.txt").write_text("1-10-0001\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1-10-0001"):
        build_manifest_librispeech(tmp_path)


def test_cli_writes_manifest(tmp_path, capsys):
    from eval.manifest_librispeech import main

    _make_corpus(tmp_path, {("1", "10"): [("0001", "SOME TEXT")]})
    out = tmp_path / "manifest.json"

    rc = main(
        ["--corpus-dir", str(tmp_path), "--out", str(out), "--limit", "1"]
    )

    assert rc == 0
    entries = json.loads(out.read_text(encoding="utf-8"))
    assert entries[0]["set"] == "librispeech"
    assert "wrote 1 entries" in capsys.readouterr().out
