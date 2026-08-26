"""Tests for ``local/make_mooncast_jsonl.py``: the input file handed to the
MoonCast baseline, built from OUR manifest so every system in the table sees
the same script and the same prompts.

MoonCast wants the speaker tag moved OUT of the text and into a ``role``
field, so the test that matters most here is the one asserting that
re-inserting the tags reproduces the other converters' text byte for byte.
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


mod = _load("make_mooncast_jsonl")
moss = _load("make_moss_ttsd_jsonl")

ROLE_TO_TAG = {"0": "[S1]", "1": "[S2]"}


def _rows(tmp_path, specs):
    fx = write_manifest(tmp_path, specs)
    return mod.build_rows(Path(fx["root"]))


def test_two_speaker_row_carries_roles_not_tags(tmp_path):
    rows, mono_ids = _rows(tmp_path, [TWO_SPK])
    assert mono_ids == []
    row = rows[0]
    assert row["window_id"] == "d2"
    assert row["dialogue"] == [
        {"role": "0", "text": "abc"},
        {"role": "1", "text": "def"},
        {"role": "0", "text": "gab"},
    ]
    assert row["role_mapping"]["0"]["ref_text"] == "abc"
    assert row["role_mapping"]["1"]["ref_text"] == "de"
    wavs = [row["role_mapping"][role]["ref_audio"] for role in ("0", "1")]
    assert wavs[0] != wavs[1]
    for wav in wavs:
        assert Path(wav).is_absolute() and Path(wav).is_file()


def test_retagged_turns_match_the_other_converters_byte_for_byte(tmp_path):
    # The whole point of building this file from our manifest: the script
    # text must be the same object for every system in the table, even
    # though this converter is the only one that strips the tags.
    fx = write_manifest(tmp_path, [TWO_SPK, ONE_SPK])
    rows, _ = mod.build_rows(Path(fx["root"]))
    moss_rows, _ = moss.build_rows(Path(fx["root"]))
    for row, moss_row in zip(rows, moss_rows):
        retagged = " ".join(
            "{} {}".format(ROLE_TO_TAG[turn["role"]], turn["text"])
            for turn in row["dialogue"]
        )
        assert retagged == moss_row["text"]


def test_both_roles_always_exist(tmp_path):
    # ``infer_with_prompt`` indexes role_mapping["0"] and ["1"]
    # unconditionally; a missing role is a KeyError inside a GPU job.
    rows, _ = _rows(tmp_path, [TWO_SPK, ONE_SPK])
    for row in rows:
        assert set(row["role_mapping"]) == {"0", "1"}


def test_monologue_row_duplicates_the_s1_prompt_into_role_one(tmp_path):
    # Thanapat, 2026-08-26.  Both prompts are injected into the
    # conditioning context, so the second one is not inert - the same voice
    # is the least distortion available.
    rows, mono_ids = _rows(tmp_path, [ONE_SPK])
    assert mono_ids == ["d1"]
    row = rows[0]
    assert row["role_mapping"]["0"] == row["role_mapping"]["1"]
    assert row["role_mapping"]["0"]["ref_text"] == "fed"
    assert all(turn["role"] == "0" for turn in row["dialogue"])


def test_more_than_two_speakers_is_an_error(tmp_path):
    # Upstream would NOT reject this: any role that is not "0" falls
    # through to speaker 1, so a third speaker would be silently merged.
    fx = write_manifest(tmp_path, [TWO_SPK])
    path = Path(fx["root"]) / "manifest.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["num_channels"] = 3
    record["channels"] = (record["channels"] * 2)[:3]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        mod.build_rows(Path(fx["root"]))


def test_an_unrepresentable_tag_is_an_error():
    with pytest.raises(ValueError, match=r"\[S3\]"):
        mod.strip_tag("[S3] hello")


def test_strip_tag_splits_role_from_text():
    assert mod.strip_tag("[S1]  hello  ") == ("0", "hello")
    assert mod.strip_tag("[S2] hi") == ("1", "hi")


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


def test_id_travels_under_window_id(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    assert rows[0]["window_id"] == "d2"


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
