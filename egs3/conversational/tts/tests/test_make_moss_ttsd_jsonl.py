"""Tests for ``local/make_moss_ttsd_jsonl.py``: the input file handed to the
MOSS-TTSD baseline, built from OUR manifest so both systems see the same
script and the same prompts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from .test_external_manifest import ONE_SPK, TWO_SPK, write_manifest

_SPEC = importlib.util.spec_from_file_location(
    "make_moss_ttsd_jsonl",
    Path(__file__).resolve().parents[1] / "local" / "make_moss_ttsd_jsonl.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _rows(tmp_path, specs):
    fx = write_manifest(tmp_path, specs)
    return mod.build_rows(Path(fx["root"]))


def test_two_speaker_row_has_one_pair_per_speaker(tmp_path):
    rows, mono_ids = _rows(tmp_path, [TWO_SPK])
    assert mono_ids == []
    row = rows[0]
    assert row["window_id"] == "d2"
    # Turn order, tagged - identical to the string the ZipVoice converter
    # emits, so the script text is the same for every system.
    assert row["text"] == "[S1] abc [S2] def [S1] gab"
    assert row["prompt_text_speaker1"] == "[S1] abc"
    assert row["prompt_text_speaker2"] == "[S2] de"
    assert Path(row["prompt_audio_speaker1"]).is_file()
    assert Path(row["prompt_audio_speaker2"]).is_file()
    assert row["prompt_audio_speaker1"] != row["prompt_audio_speaker2"]
    # Absolute: we send no base_path, so their loader must not have to
    # resolve anything.
    assert Path(row["prompt_audio_speaker1"]).is_absolute()
    assert "base_path" not in row


def test_monologue_row_carries_speaker_one_only(tmp_path):
    rows, mono_ids = _rows(tmp_path, [ONE_SPK])
    assert mono_ids == ["d1"]
    row = rows[0]
    assert "prompt_audio_speaker2" not in row
    assert "prompt_text_speaker2" not in row
    assert row["prompt_text_speaker1"] == "[S1] fed"
    assert "[S2]" not in row["text"]


def test_id_key_is_window_id_not_id(tmp_path):
    # Their ``_make_output_record`` OVERWRITES ``id`` with the line number,
    # so the id we send back has to travel under a different key.
    rows, _ = _rows(tmp_path, [TWO_SPK])
    assert "id" not in rows[0]
    assert rows[0]["window_id"] == "d2"


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


def test_shards_partition_the_rows_in_order(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK, ONE_SPK])
    paths = mod.write_shards(rows, tmp_path / "input.jsonl", num_shards=2)
    assert [p.name for p in paths] == ["input.00.jsonl", "input.01.jsonl"]
    seen = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            seen.append(json.loads(line)["window_id"])
    assert seen == ["d2", "d1"]


def test_one_shard_writes_the_plain_file(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    paths = mod.write_shards(rows, tmp_path / "input.jsonl", num_shards=1)
    assert [p.name for p in paths] == ["input.jsonl"]
    assert json.loads(paths[0].read_text())["window_id"] == "d2"
