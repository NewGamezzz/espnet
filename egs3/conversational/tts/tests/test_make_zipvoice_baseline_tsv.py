"""Tests for ``local/make_zipvoice_baseline_tsv.py``: the input file handed
to the ZipVoice-Dialog baseline, built from OUR manifest so both systems see
the same script and the same prompts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from .test_external_manifest import ONE_SPK, TWO_SPK, write_manifest

_SPEC = importlib.util.spec_from_file_location(
    "make_zipvoice_baseline_tsv",
    Path(__file__).resolve().parents[1] / "local" / "make_zipvoice_baseline_tsv.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _rows(tmp_path, specs):
    fx = write_manifest(tmp_path, specs)
    return mod.build_rows(Path(fx["root"]))


def test_two_speaker_row_is_their_split_format(tmp_path):
    rows, mono_ids = _rows(tmp_path, [TWO_SPK])
    assert mono_ids == []
    fields = rows[0].split("\t")
    assert len(fields) == 6
    wid, text_a, text_b, wav_a, wav_b, text = fields
    assert wid == "d2"
    assert (text_a, text_b) == ("abc", "de")
    assert wav_a != wav_b
    assert Path(wav_a).is_file() and Path(wav_b).is_file()
    # Turn order, tagged - and their loader asserts the [S1] start.
    assert text == "[S1] abc [S2] def [S1] gab"


def test_monologue_row_duplicates_the_single_prompt(tmp_path):
    rows, mono_ids = _rows(tmp_path, [ONE_SPK])
    assert mono_ids == ["d1"]
    wid, text_a, text_b, wav_a, wav_b, text = rows[0].split("\t")
    assert (text_a, text_b) == ("fed", "fed")
    assert wav_a == wav_b
    assert "[S2]" not in text


def test_missing_prompt_audio_is_an_error(tmp_path):
    fx = write_manifest(tmp_path, [TWO_SPK])
    Path(fx["root"], "prompt/d2_ch1.wav").unlink()
    with pytest.raises(FileNotFoundError):
        mod.build_rows(Path(fx["root"]))


def test_identical_archive_rows_produce_no_diff(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    archive = tmp_path / "test.tsv"
    archive.write_text(
        "d2\tabc\tde\tp/a.wav\tp/b.wav\t[S1] abc [S2] def [S1] gab\n", encoding="utf-8"
    )
    assert mod.diff_against_archive(rows, archive) == []


def test_archive_differences_are_reported_per_field(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    archive = tmp_path / "test.tsv"
    archive.write_text(
        "d2\tabc\tDIFFERENT\tp/a.wav\tp/b.wav\t[S1] abc [S2] def\n", encoding="utf-8"
    )
    fields = {field for _, field, _, _ in mod.diff_against_archive(rows, archive)}
    assert fields == {"text", "prompt_text"}


def test_rows_not_in_the_archive_are_skipped(tmp_path):
    rows, _ = _rows(tmp_path, [TWO_SPK])
    archive = tmp_path / "test.tsv"
    archive.write_text("other\ta\tb\tp/a.wav\tp/b.wav\t[S1] x\n", encoding="utf-8")
    assert mod.diff_against_archive(rows, archive) == []
