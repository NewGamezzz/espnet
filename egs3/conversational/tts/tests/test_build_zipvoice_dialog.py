"""Tests for local/build_zipvoice_dialog_testset.py: the ZipVoice-Dialog
test-en tarball reformatted into the training-style external manifest.

Fixture-based: a fabricated ``dialog_testset/en`` tree (``test.tsv``, stereo
prompt wavs with ONE active track, stereo dual-track ground-truth wavs) that
reproduces the real archive's conventions, including its edge cases:
single-speaker dialogues, consecutive same-speaker tags, an empty segment,
and a prompt-A track that is L for one session and R for another.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.local.build_zipvoice_dialog_testset import (
    active_track,
    build,
    main,
    parse_dialogue_text,
)
from egs3.conversational.tts.src.external_testset import load_external_manifest

from .conftest import EXT_TOKENS

SR = 24000


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
class TestParseDialogueText:
    def test_tags_become_channels_in_order(self):
        turns, dropped = parse_dialogue_text("[S1] abc [S2] def [S1] gab")
        assert turns == [(0, "abc"), (1, "def"), (0, "gab")]
        assert dropped == 0

    def test_consecutive_same_speaker_tags_stay_separate_turns(self):
        turns, _ = parse_dialogue_text("[S1] abc [S2] def [S1] ga [S1] b")
        assert turns == [(0, "abc"), (1, "def"), (0, "ga"), (0, "b")]

    def test_empty_segments_are_dropped_and_counted(self):
        turns, dropped = parse_dialogue_text("[S1] abc [S2] [S1] def [S2]  ")
        assert turns == [(0, "abc"), (0, "def")]
        assert dropped == 2

    def test_single_speaker_dialogue(self):
        turns, _ = parse_dialogue_text("[S1] abc def")
        assert turns == [(0, "abc def")]

    def test_text_before_the_first_tag_is_rejected(self):
        with pytest.raises(ValueError, match="before the first"):
            parse_dialogue_text("abc [S1] def")

    def test_unknown_tag_is_rejected(self):
        with pytest.raises(ValueError, match="S3"):
            parse_dialogue_text("[S1] abc [S3] def")


class TestActiveTrack:
    def test_picks_the_louder_channel(self):
        loud = 0.3 * np.sin(np.linspace(0, 200, SR))
        quiet = 1e-4 * np.random.RandomState(0).randn(SR)
        assert active_track(np.stack([loud, quiet], axis=1)) == 0
        assert active_track(np.stack([quiet, loud], axis=1)) == 1

    def test_mono_input_is_track_zero(self):
        assert active_track(np.zeros((SR, 1))) == 0


# --------------------------------------------------------------------------- #
# fabricated archive
# --------------------------------------------------------------------------- #
def _tone(seconds: float, freq: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _floor(n: int, seed: int) -> np.ndarray:
    return (1e-4 * np.random.RandomState(seed).randn(n)).astype(np.float32)


def _write_stereo(path: Path, tracks: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack(tracks, axis=1), SR, subtype="PCM_16")


def _prompt(path: Path, seconds: float, active: int) -> None:
    n = int(round(seconds * SR))
    tracks = [_floor(n, 1), _floor(n, 2)]
    tracks[active] = _tone(seconds, 440.0)
    _write_stereo(path, tracks)


def _gt(path: Path, seconds: float, first: int, silent_other: bool = False) -> None:
    """Dual-track ground truth: channel ``first`` speaks from 0 s, the other
    from half-way (or never)."""
    n = int(round(seconds * SR))
    tracks = [_floor(n, 3), _floor(n, 4)]
    tracks[first][: n // 2] = _tone(seconds / 2, 300.0)[: n // 2]
    if not silent_other:
        tracks[1 - first][n // 2 :] = _tone(seconds / 2, 500.0)[: n - n // 2]
    _write_stereo(path, tracks)


# session X: prompt A active on L (0); session Y: prompt A active on R (1).
ROWS = [
    # (id, prompt_text_A, prompt_text_B, prompt_A, prompt_B, text, gt_first_channel)
    (
        "X_0-000",
        "abc",
        "de",
        "X_0-001-A.wav",
        "X_0-001-B.wav",
        "[S1] abc [S2] def [S1] gab",
        0,
    ),
    (
        "X_0-001",
        "abc",
        "de",
        "X_0-001-A.wav",
        "X_0-001-B.wav",
        "[S1] fed [S1] cab [S2] ab",
        0,
    ),
    (
        "Y_0-000",
        "fed",
        "cab",
        "Y_0-002-A.wav",
        "Y_0-002-B.wav",
        "[S1] abc [S2] [S2] bad",
        1,
    ),
    ("Y_0-001", "fed", "cab", "Y_0-002-A.wav", "Y_0-002-B.wav", "[S1] cab bad", 1),
    # transcript says S1 first but the OTHER channel has the first onset
    ("Y_0-002", "fed", "cab", "Y_0-002-A.wav", "Y_0-002-B.wav", "[S1] abc [S2] def", 0),
]
PROMPTS = {
    "X_0-001-A.wav": 0,
    "X_0-001-B.wav": 1,
    "Y_0-002-A.wav": 1,
    "Y_0-002-B.wav": 0,
}


def write_archive(tmp_path: Path) -> dict:
    src = tmp_path / "dialog_testset" / "en"
    for name, active in PROMPTS.items():
        _prompt(src / "prompt_wavs" / name, 1.5, active)
    lines = []
    for wid, pa, pb, wa, wb, text, first in ROWS:
        _gt(
            src / "ground_truth_wavs" / f"{wid}.wav",
            4.0,
            first,
            silent_other=(wid == "Y_0-001"),
        )
        lines.append(
            "\t".join(
                [
                    wid,
                    pa,
                    pb,
                    f"download/dialog_testset/en/prompt_wavs/{wa}",
                    f"download/dialog_testset/en/prompt_wavs/{wb}",
                    text,
                ]
            )
        )
    (src / "test.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    return {"src": src, "vocab": vocab, "out": tmp_path / "zipvoice-dialog-test-en"}


@pytest.fixture
def archive(tmp_path):
    return write_archive(tmp_path)


def _lines(out: Path) -> dict[str, dict]:
    text = (out / "manifest.jsonl").read_text("utf-8")
    return {
        json.loads(ln)["window_id"]: json.loads(ln) for ln in text.splitlines() if ln
    }


class TestBuild:
    def test_manifest_has_one_record_per_row_in_tsv_order(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        recs = _lines(archive["out"])
        assert list(recs) == [r[0] for r in ROWS]
        assert (archive["out"] / "build_meta.json").is_file()

    def test_two_speaker_record_shape(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        rec = _lines(archive["out"])["X_0-000"]
        assert rec["num_channels"] == 2
        assert rec["session_id"] == "X_0"
        assert [(t["channel"], t["speaker"], t["text"]) for t in rec["turns"]] == [
            (0, "S1", "abc"),
            (1, "S2", "def"),
            (0, "S1", "gab"),
        ]
        assert rec["channels"][0]["prompt_text"] == "abc"
        assert rec["channels"][1]["prompt_text"] == "de"
        assert rec["channels"][0]["prompt_wav"] == "prompt/X_0-001-A.wav"
        assert rec["channels"][0]["gt_wav"] == "gt/X_0-000_ch0.wav"
        assert rec["channels"][1]["gt_wav"] == "gt/X_0-000_ch1.wav"

    def test_single_speaker_row_is_a_one_channel_record(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        rec = _lines(archive["out"])["Y_0-001"]
        assert rec["num_channels"] == 1
        assert rec["turns"] == [{"channel": 0, "speaker": "S1", "text": "cab bad"}]
        assert len(rec["channels"]) == 1
        assert rec["channels"][0]["prompt_wav"] == "prompt/Y_0-002-A.wav"
        assert rec["channels"][0]["gt_wav"] == "gt/Y_0-001_ch0.wav"
        assert not (archive["out"] / "gt/Y_0-001_ch1.wav").exists()

    def test_prompts_are_extracted_as_the_active_track(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        for name, active in PROMPTS.items():
            mono, sr = sf.read(str(archive["out"] / "prompt" / name), always_2d=True)
            assert sr == SR and mono.shape[1] == 1
            stereo, _ = sf.read(
                str(archive["src"] / "prompt_wavs" / name), always_2d=True
            )
            assert np.array_equal(mono[:, 0], stereo[:, active])
        meta = json.loads((archive["out"] / "build_meta.json").read_text("utf-8"))
        assert meta["prompt_tracks"] == PROMPTS

    def test_ground_truth_channels_follow_the_prompt_a_track(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        # Session Y: prompt A is active on R, so S1 = R track (index 1).
        stereo, _ = sf.read(
            str(archive["src"] / "ground_truth_wavs/Y_0-000.wav"), always_2d=True
        )
        ch0, _ = sf.read(str(archive["out"] / "gt/Y_0-000_ch0.wav"), always_2d=True)
        ch1, _ = sf.read(str(archive["out"] / "gt/Y_0-000_ch1.wav"), always_2d=True)
        assert np.array_equal(ch0[:, 0], stereo[:, 1])
        assert np.array_equal(ch1[:, 0], stereo[:, 0])
        # Session X: prompt A active on L, so S1 = L track (index 0).
        stereo, _ = sf.read(
            str(archive["src"] / "ground_truth_wavs/X_0-000.wav"), always_2d=True
        )
        ch0, _ = sf.read(str(archive["out"] / "gt/X_0-000_ch0.wav"), always_2d=True)
        assert np.array_equal(ch0[:, 0], stereo[:, 0])

    def test_edge_cases_are_recorded_in_build_meta(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        meta = json.loads((archive["out"] / "build_meta.json").read_text("utf-8"))
        assert meta["n_dialogues"] == 5
        assert meta["n_single_speaker"] == 1
        assert meta["single_speaker_ids"] == ["Y_0-001"]
        assert meta["n_dropped_empty_segments"] == 1
        assert meta["dropped_empty_segments"] == [{"window_id": "Y_0-000", "count": 1}]
        assert meta["consecutive_same_speaker_ids"] == ["X_0-001"]
        anomalies = {a["window_id"]: a for a in meta["anomalies"]}
        # Y_0-002: the transcript's S1 is the prompt-A track (R) but L
        # speaks first -> flagged, mapping kept (prompt rule wins).
        assert anomalies["Y_0-002"]["kind"] == "first_onset_not_s1"
        assert anomalies["Y_0-002"]["lead_sec"] > 0
        # Y_0-001 is single-speaker: its unused GT track is silent, which is
        # expected there, so it is NOT an anomaly.
        assert "Y_0-001" not in anomalies

    def test_silent_ground_truth_channel_on_a_two_speaker_row_is_flagged(
        self, tmp_path
    ):
        fx = write_archive(tmp_path)
        _gt(fx["src"] / "ground_truth_wavs/X_0-000.wav", 4.0, 0, silent_other=True)
        build(fx["src"], fx["out"], fx["vocab"])
        meta = json.loads((fx["out"] / "build_meta.json").read_text("utf-8"))
        kinds = {(a["window_id"], a["kind"]) for a in meta["anomalies"]}
        assert ("X_0-000", "silent_gt_channel") in kinds

    def test_dry_run_loads_every_record_through_the_real_loader(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        meta = json.loads((archive["out"] / "build_meta.json").read_text("utf-8"))
        records = load_external_manifest(
            archive["out"] / "manifest.jsonl", archive["vocab"]
        )
        assert [r.dialogue_id for r in records] == [r[0] for r in ROWS]
        assert meta["loader_dry_run"]["n_records"] == 5
        assert meta["loader_dry_run"]["n_channels"] == {"1": 1, "2": 4}
        assert meta["loader_dry_run"]["total_gt_hours"] == pytest.approx(5 * 4.0 / 3600)

    def test_characters_outside_the_charset_are_reported(self, tmp_path):
        fx = write_archive(tmp_path)
        tsv = fx["src"] / "test.tsv"
        rows = tsv.read_text("utf-8").splitlines()
        rows[0] = rows[0].replace("[S1] abc", '[S1] "abc+')
        tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        build(fx["src"], fx["out"], fx["vocab"])
        meta = json.loads((fx["out"] / "build_meta.json").read_text("utf-8"))
        assert meta["loader_dry_run"]["chars_dropped_by_normalization"] == {
            '"': 1,
            "+": 1,
        }

    def test_archive_md5_is_recorded_when_given(self, archive, tmp_path):
        tarball = tmp_path / "dialog_testset.tar.gz"
        tarball.write_bytes(b"not really a tarball")
        build(archive["src"], archive["out"], archive["vocab"], archive=tarball)
        meta = json.loads((archive["out"] / "build_meta.json").read_text("utf-8"))
        import hashlib

        assert (
            meta["archive"]["md5"] == hashlib.md5(b"not really a tarball").hexdigest()
        )
        assert meta["archive"]["path"] == str(tarball)

    def test_source_file_list_is_pinned(self, archive):
        build(archive["src"], archive["out"], archive["vocab"])
        meta = json.loads((archive["out"] / "build_meta.json").read_text("utf-8"))
        assert meta["source_files"]["tsv_md5"]
        assert set(meta["source_files"]["prompt_wavs"]) == set(PROMPTS)
        assert len(meta["source_files"]["ground_truth_wavs"]) == 5

    def test_prompt_b_on_the_same_track_as_prompt_a_is_an_error(self, tmp_path):
        fx = write_archive(tmp_path)
        _prompt(fx["src"] / "prompt_wavs/X_0-001-B.wav", 1.5, active=0)
        with pytest.raises(ValueError, match="same track"):
            build(fx["src"], fx["out"], fx["vocab"])

    def test_cli(self, archive):
        main(
            [
                "--src",
                str(archive["src"]),
                "--out",
                str(archive["out"]),
                "--token_list",
                str(archive["vocab"]),
            ]
        )
        assert (archive["out"] / "manifest.jsonl").is_file()
