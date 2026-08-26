"""Tests for ``local/make_firered_jsonl.py``: the input file handed to the
FireRedTTS-2 baseline, built from OUR manifest so every system in the table
sees the same script and the same prompts.

FireRedTTS-2 takes a LIST of tagged turns rather than one merged string, so
the test that matters most here is the one asserting the joined list is
byte-identical to what the ZipVoice and MOSS-TTSD converters emit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from .test_external_manifest import ONE_SPK, TWO_SPK, write_manifest


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "local" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load("make_firered_jsonl")
moss = _load("make_moss_ttsd_jsonl")


def _rows(tmp_path, specs):
    fx = write_manifest(tmp_path, specs)
    return mod.build_rows(Path(fx["root"]))


def test_two_speaker_row_is_a_tagged_turn_list(tmp_path):
    rows, mono_ids = _rows(tmp_path, [TWO_SPK])
    assert mono_ids == []
    row = rows[0]
    assert row["window_id"] == "d2"
    assert row["text_list"] == ["[S1] abc", "[S2] def", "[S1] gab"]
    assert row["prompt_text_list"] == ["[S1] abc", "[S2] de"]
    assert len(row["prompt_wav_list"]) == 2
    assert row["prompt_wav_list"][0] != row["prompt_wav_list"][1]
    for wav in row["prompt_wav_list"]:
        assert Path(wav).is_absolute() and Path(wav).is_file()


def test_joined_turns_match_the_other_converters_byte_for_byte(tmp_path):
    # The whole point of building this file from our manifest: the script
    # text must be the same object for every system in the table.
    fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
    rows, _ = mod.build_rows(Path(fx["root"]))
    moss_rows, _ = moss.build_rows(Path(fx["root"]))
    for row, moss_row in zip(rows, moss_rows):
        assert " ".join(row["text_list"]) == moss_row["text"]


def test_speaker_tag_occupies_exactly_the_first_four_characters(tmp_path):
    # ``generate_dialogue`` reads ``text[:4]`` and asserts it is one of
    # ``[S1]``..``[S4]``; ``process_text_list`` does the same.  A tag one
    # character off is an assertion failure 280 rows into a GPU job.
    rows, _ = _rows(tmp_path, [TWO_SPK])
    row = rows[0]
    for text in row["text_list"] + row["prompt_text_list"]:
        assert text[:4] in {"[S1]", "[S2]", "[S3]", "[S4]"}


def test_monologue_row_carries_speaker_one_only(tmp_path):
    rows, mono_ids = _rows(tmp_path, [ONE_SPK])
    assert mono_ids == ["d1"]
    row = rows[0]
    assert row["prompt_text_list"] == ["[S1] fed"]
    assert len(row["prompt_wav_list"]) == 1
    assert all(text.startswith("[S1]") for text in row["text_list"])


def test_id_travels_under_window_id(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    assert rows[0]["window_id"] == "d2"


def test_more_than_four_speakers_is_an_error(tmp_path):
    # Their tag vocabulary stops at [S4].  We never hit this on a 2-speaker
    # set, but a 3-speaker set is already in the framework's future.
    fx = write_manifest(tmp_path, [TWO_SPK])
    path = Path(fx["root"]) / "manifest.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["num_channels"] = 5
    record["channels"] = record["channels"] * 3
    record["channels"] = record["channels"][:5]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="S4"):
        mod.build_rows(Path(fx["root"]))


def test_missing_prompt_audio_is_an_error(tmp_path):
    fx = write_manifest(tmp_path, [TWO_SPK])
    Path(fx["root"], "prompt/d2_ch1.wav").unlink()
    with pytest.raises(FileNotFoundError):
        mod.build_rows(Path(fx["root"]))


def test_text_not_starting_with_s1_is_an_error(tmp_path):
    fx = write_manifest(tmp_path, [TWO_SPK])
    path = Path(fx["root"]) / "manifest.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["turns"][0]["speaker"] = "S2"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[S1\]"):
        mod.build_rows(Path(fx["root"]))


def test_shards_are_contiguous_and_cover_every_row(tmp_path):
    rows = [{"window_id": f"d{i}"} for i in range(7)]
    out = tmp_path / "input.jsonl"
    paths = mod.write_shards(rows, out, 3)
    assert [p.name for p in paths] == [
        "input.00.jsonl",
        "input.01.jsonl",
        "input.02.jsonl",
    ]
    got = []
    for path in paths:
        got += [json.loads(line) for line in path.read_text().splitlines()]
    assert got == rows


def test_single_shard_writes_one_file(tmp_path):
    out = tmp_path / "input.jsonl"
    assert mod.write_shards([{"window_id": "d0"}], out, 1) == [out]
    assert json.loads(out.read_text().strip())["window_id"] == "d0"
