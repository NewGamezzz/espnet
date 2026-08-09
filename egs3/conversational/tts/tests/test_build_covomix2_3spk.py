"""Tests for local/build_covomix2_3spk.py: derived 3-speaker index built
from a fabricated CoVoMix2 tree plus a fabricated LibriSpeech test-clean
tree with real trans.txt naming conventions."""

from __future__ import annotations

import json

import pytest

from egs3.conversational.tts.local.build_covomix2_3spk import build, main, probe_sec
from egs3.conversational.tts.src.external_testset import load_covomix2_testset

from .test_external_testset import DIALOGUES, PROMPT_SR, _write_flac, _write_testset


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
    # Native pairs: "000" is (spk1=1 @1.0s, spk2=2 @2.0s), so its per-dialogue
    # band is [0.75, 2.5]s; "001" is (spk1=3 @1.5s, spk2=4 @1.0s), band
    # [0.75, 1.875]s.  Speaker 8's utterance (2.2s) sits inside "000"'s band
    # but outside "001"'s, forcing "001" to pick speaker 7 - whose sole
    # utterance is "ABC Def", so the lowercasing requirement (trans.txt is
    # ALL-CAPS/mixed-case; the index must store lowercase like the native
    # spk1/spk2 rows) is deterministically exercised.  Speaker 9's
    # utterance (5.0s) is outside both per-dialogue bands and must never be
    # chosen.
    _libri_tree(
        ts["librispeech_root"],
        {
            "7": [("1", "0001", "ABC Def", 1.2)],
            "8": [("2", "0001", "chad face", 2.2)],
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
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        # This fixture's two dialogues both have an in-band candidate (7 or
        # 8), so neither should have needed the fallback - asserted here so
        # the per-dialogue-band check below can require it unconditionally.
        assert meta["n_fallback"] == 0
        for e in entries:
            spk3 = e["audio_prompt_spk3"].split("/")[1]
            spk1 = e["audio_prompt_spk1"].split("/")[1]
            spk2 = e["audio_prompt_spk2"].split("/")[1]
            assert spk3 not in {spk1, spk2}
            assert spk3 != "9"  # out-of-band speaker never eligible
            d1 = probe_sec(tree["librispeech_root"] / e["audio_prompt_spk1"])
            d2 = probe_sec(tree["librispeech_root"] / e["audio_prompt_spk2"])
            lo, hi = 0.75 * min(d1, d2), 1.25 * max(d1, d2)
            sec = probe_sec(tree["librispeech_root"] / e["audio_prompt_spk3"])
            assert lo <= sec <= hi

    def test_transcription_matches_trans_txt(self, tree, tmp_path):
        out = tmp_path / "out3spk"
        build(tree["testset_root"], tree["librispeech_root"], out, seed=0)
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        # trans.txt is ALL-CAPS/mixed-case by convention; the index must
        # store the lowercased form, matching native spk1/spk2 rows.
        by_utt = {
            "7-1-0001": "abc def",
            "8-2-0001": "chad face",
        }
        for e in entries:
            utt_id = e["audio_prompt_spk3"].rsplit("/", 1)[-1].removesuffix(".flac")
            assert e["audio_prompt_spk3_transcription"] == by_utt[utt_id]

    def test_fallback_used_when_per_dialogue_band_has_no_candidate(self, tmp_path):
        # A third dialogue whose native pair is very short (0.1, 0.1 s)
        # makes the per-dialogue band [0.075, 0.125] s - none of speakers
        # 7/8/9's utterances (1.2/1.8/1.5/5.0 s) fall in it, so this entry
        # must fall back to the global [p10, p90] band.
        dialogues = dict(DIALOGUES)
        dialogues["002"] = (
            ["abc", "def"],
            [
                ("test-clean/10/10/x.flac", "abc", 0.1),
                ("test-clean/11/11/y.flac", "def", 0.1),
            ],
        )
        ts = _write_testset(tmp_path, dialogues=dialogues)
        _libri_tree(
            ts["librispeech_root"],
            {
                "7": [("1", "0001", "abc def", 1.2), ("1", "0002", "bead", 1.8)],
                "8": [("2", "0001", "chad face", 1.5)],
                "9": [("3", "0001", "gaff", 5.0)],
            },
        )
        out = tmp_path / "out3spk_fallback"
        meta = build(ts["testset_root"], ts["librispeech_root"], out, seed=0)
        assert meta["n_fallback"] == 1
        fb_lo, fb_hi = meta["global_fallback_band_sec"]
        entries = json.loads(
            (out / "dailydialog-dialogue.json").read_text(encoding="utf-8")
        )
        entry = next(e for e in entries if e["key"] == "002")
        spk3 = entry["audio_prompt_spk3"].split("/")[1]
        assert spk3 != "9"  # out-of-band speaker never eligible
        sec = probe_sec(ts["librispeech_root"] / entry["audio_prompt_spk3"])
        assert fb_lo <= sec <= fb_hi

    def test_build_meta_has_band_and_case_keys(self, tree, tmp_path):
        out = tmp_path / "out3spk"
        meta = build(tree["testset_root"], tree["librispeech_root"], out, seed=0)
        assert meta["seed"] == 0
        assert meta["n_dialogues"] == 2
        assert meta["source_index"] == str(
            tree["testset_root"] / "dailydialog-dialogue.json"
        )
        assert meta["transcription_case"] == "lower"
        assert meta["band_rule"] == (
            "per-dialogue [0.75*min, 1.25*max], fallback global [p10, p90]"
        )
        assert len(meta["global_fallback_band_sec"]) == 2
        assert meta["n_fallback"] == 0
        assert "duration_band_sec" not in meta
        on_disk = json.loads((out / "build_meta.json").read_text(encoding="utf-8"))
        assert on_disk == meta

    def test_deterministic_for_a_seed(self, tree, tmp_path):
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
