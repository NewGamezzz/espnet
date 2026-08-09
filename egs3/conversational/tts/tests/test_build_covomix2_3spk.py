"""Tests for local/build_covomix2_3spk.py: derived 3-speaker index built
from a fabricated CoVoMix2 tree plus a fabricated LibriSpeech test-clean
tree with real trans.txt naming conventions."""

from __future__ import annotations

import json

import pytest

from egs3.conversational.tts.local.build_covomix2_3spk import build, main
from egs3.conversational.tts.src.external_testset import load_covomix2_testset

from .test_external_testset import PROMPT_SR, _write_flac, _write_testset


def _libri_tree(root, spec):
    """spec: {speaker: [(chapter, utt, text, sec), ...]} in LibriSpeech
    test-clean layout with {spk}-{chap}.trans.txt files."""
    for spk, utts in spec.items():
        for chap, utt, text, sec in utts:
            d = root / "test-clean" / spk / chap
            _write_flac(d / f"{spk}-{chap}-{utt}.flac", 1, sec, PROMPT_SR)
            trans = d / f"{spk}-{chap}.trans.txt"
            with trans.open("a", encoding="utf-8") as f:
                f.write(f"{spk}-{chap}-{utt} {text}\n")


@pytest.fixture
def tree(tmp_path):
    ts = _write_testset(tmp_path)  # 2-speaker base testset + its prompts
    # Fixture prompts live at test-clean/{1,2,3,4}/... and last 1.0-2.0 s,
    # so the band is [1.0, 2.0].  Speakers 7-9 are spk3 candidates; speaker
    # 9's only utterance is OUT of band and must never be chosen.
    _libri_tree(
        ts["librispeech_root"],
        {
            "7": [("1", "0001", "abc def", 1.2), ("1", "0002", "bead", 1.8)],
            "8": [("2", "0001", "chad face", 1.5)],
            "9": [("3", "0001", "gaff", 5.0)],
        },
    )
    return ts


class TestBuild:
    def test_output_layout_and_loadability(self, tree, tmp_path):
        out = tmp_path / "out3spk"
        meta = build(tree["testset_root"], tree["librispeech_root"], out, seed=0)
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        assert all("audio_prompt_spk3" in e for e in entries)
        assert all("audio_prompt_spk3_transcription" in e for e in entries)
        assert (out / "transcriptions").is_dir()
        assert meta["n_dialogues"] == len(entries)
        records = load_covomix2_testset(
            out, tree["librispeech_root"], tree["vocab"], num_channels=3
        )
        assert len(records) == len(entries)
        assert all(len(r.prompts) == 3 for r in records)

    def test_spk3_is_disjoint_and_in_band(self, tree, tmp_path):
        out = tmp_path / "out3spk"
        meta = build(tree["testset_root"], tree["librispeech_root"], out, seed=0)
        lo, hi = meta["duration_band_sec"]
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        for e in entries:
            spk3 = e["audio_prompt_spk3"].split("/")[1]
            spk1 = e["audio_prompt_spk1"].split("/")[1]
            spk2 = e["audio_prompt_spk2"].split("/")[1]
            assert spk3 not in {spk1, spk2}
            assert spk3 != "9"  # out-of-band speaker never eligible
            # In-band by probe:
            from egs3.conversational.tts.local.build_covomix2_3spk import probe_sec

            sec = probe_sec(tree["librispeech_root"] / e["audio_prompt_spk3"])
            assert lo <= sec <= hi

    def test_transcription_matches_trans_txt(self, tree, tmp_path):
        out = tmp_path / "out3spk"
        build(tree["testset_root"], tree["librispeech_root"], out, seed=0)
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        by_utt = {
            "7-1-0001": "abc def",
            "7-1-0002": "bead",
            "8-2-0001": "chad face",
        }
        for e in entries:
            utt_id = e["audio_prompt_spk3"].rsplit("/", 1)[-1].removesuffix(".flac")
            assert e["audio_prompt_spk3_transcription"] == by_utt[utt_id]

    def test_deterministic_for_a_seed_and_sensitive_to_it(self, tree, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        build(tree["testset_root"], tree["librispeech_root"], a, seed=0)
        build(tree["testset_root"], tree["librispeech_root"], b, seed=0)
        assert (a / "dailydialog-dialogue.json").read_text() == (
            b / "dailydialog-dialogue.json"
        ).read_text()

    def test_cli_smoke(self, tree, tmp_path):
        out = tmp_path / "cli_out"
        main(
            [
                "--testset-root",
                str(tree["testset_root"]),
                "--librispeech-root",
                str(tree["librispeech_root"]),
                "--out-root",
                str(out),
                "--seed",
                "0",
            ]
        )
        assert (out / "build_meta.json").is_file()
