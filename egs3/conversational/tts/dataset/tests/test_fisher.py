"""Fisher manifest parsing, text cleaning, and A/B -> stereo merge machinery."""

import gzip
import json
import string
import subprocess
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.preprocessing import fisher
from egs3.conversational.tts.dataset.preprocessing.sssd import Supervision

from .conftest import REPO_ROOT, write_flac  # noqa: F401  (sys.path setup)


def sup(text, start=0.0, duration=1.0, channel=0, speaker="s0"):
    return Supervision(
        id=f"u-{start}",
        recording_id="fe_03_00001",
        channel=channel,
        start=start,
        duration=duration,
        text=text,
        speaker=speaker,
    )


def test_clean_keeps_plain_text():
    res = fisher.clean_fisher_text("and i generally prefer")
    assert res == fisher.CleanResult("and i generally prefer", False)


def test_clean_strips_event_tags_keeping_words():
    res = fisher.clean_fisher_text("well [laughter] you know [noise] right")
    assert res == fisher.CleanResult("well you know right", False)


def test_clean_unwraps_unclear_markers():
    res = fisher.clean_fisher_text("i think (( yeah maybe )) so")
    assert res == fisher.CleanResult("i think yeah maybe so", False)


def test_clean_maps_underscore_to_space():
    res = fisher.clean_fisher_text("watching t._v. tonight")
    assert res == fisher.CleanResult("watching t. v. tonight", False)


def test_clean_tag_only_utterance_is_benign_empty():
    res = fisher.clean_fisher_text("[laughter]")
    assert res == fisher.CleanResult("", False)
    res = fisher.clean_fisher_text("[sigh] [lipsmack]")
    assert res == fisher.CleanResult("", False)


def test_clean_empty_unclear_is_unintelligible():
    assert fisher.clean_fisher_text("(( ))").unintelligible
    assert fisher.clean_fisher_text("(( [noise] ))").unintelligible


def test_clean_foreign_and_unrepresentable_are_unintelligible():
    assert fisher.clean_fisher_text("<german (( ja wohl )) >").unintelligible
    assert fisher.clean_fisher_text("call me at 1 800").unintelligible
    assert fisher.clean_fisher_text("a. t. & t.").unintelligible
    assert fisher.clean_fisher_text("uh *huh*").unintelligible


def test_clean_double_bracket_skip_tag():
    # "[[skip]" appears in the corpus; the tag regex must consume it whole.
    res = fisher.clean_fisher_text("[[skip] and then")
    assert res == fisher.CleanResult("and then", False)


def test_clean_supervisions_partitions_and_collects_spans():
    sups = [
        sup("hello there", start=0.0, duration=2.0),
        sup("[laughter]", start=2.0, duration=0.5),          # benign drop
        sup("(( ))", start=3.0, duration=1.0),                # span
        sup("so [noise] anyway", start=5.0, duration=2.0),    # kept, cleaned
        sup("call 911 now", start=8.0, duration=1.0),         # span (digits)
    ]
    kept, spans, n_benign = fisher.clean_fisher_supervisions(sups)
    assert [s.text for s in kept] == ["hello there", "so anyway"]
    # cleaned supervisions keep their timing/channel/speaker fields
    assert kept[1].start == 5.0 and kept[1].duration == 2.0
    assert spans == [(3.0, 4.0), (8.0, 9.0)]
    assert n_benign == 1


def write_fisher_recordings(
    path: Path, recs: dict[str, float], *, swap_sources=False, mutate=None
) -> None:
    """recs: id -> duration_s. Real corpus layout/fields; sources point at a
    machine-specific scratch prefix on purpose. ``mutate`` edits each record
    dict before writing (for malformed-input tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for rid, duration in sorted(recs.items()):
            shard = rid.split("_")[-1][:3]
            sources = [
                {
                    "type": "file",
                    "channels": [0],
                    "source": f"/scratch/elsewhere/fisher_wavs_sidon_24k/{shard}/{rid}-A.flac",
                },
                {
                    "type": "file",
                    "channels": [1],
                    "source": f"/scratch/elsewhere/fisher_wavs_sidon_24k/{shard}/{rid}-B.flac",
                },
            ]
            if swap_sources:
                sources = sources[::-1]
            record = {
                "id": rid,
                "sources": sources,
                "sampling_rate": 24000,
                "num_samples": round(duration * 24000),
                "duration": duration,
                "channel_ids": [0, 1],
            }
            if mutate:
                mutate(record)
            f.write(json.dumps(record) + "\n")


def test_load_fisher_recordings_merged_relpath(tmp_path):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 608.484})
    recs = fisher.load_fisher_recordings(path)
    rec = recs["fe_03_00001"]
    # points at the MERGED stereo flac, sharded like the source; never the
    # scratch-absolute mono paths
    assert rec.audio_relpath == "000/fe_03_00001.flac"
    assert (rec.sample_rate, rec.num_channels) == (24000, 2)
    assert rec.duration == 608.484


def test_load_fisher_recordings_source_order_is_channel_driven(tmp_path):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 10.0}, swap_sources=True)
    rec = fisher.load_fisher_recordings(path)["fe_03_00001"]
    a, b = fisher.channel_source_relpaths(rec)
    assert a == "fisher_wavs_sidon_24k/000/fe_03_00001-A.flac"
    assert b == "fisher_wavs_sidon_24k/000/fe_03_00001-B.flac"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["sources"].pop(), "2 sources"),
        (lambda r: r["sources"][0]["channels"].append(1), "single-channel"),
        (
            lambda r: r["sources"][1].__setitem__("channels", [0]),
            "channels",
        ),
        (
            lambda r: r["sources"][0].__setitem__(
                "source", "/x/000/fe_03_00001-B.flac"
            ),
            "-A.flac",
        ),
        (
            lambda r: r["sources"][1].__setitem__(
                "source", "/x/999/fe_03_00001-B.flac"
            ),
            "shard",
        ),
    ],
)
def test_load_fisher_recordings_rejects_malformed(tmp_path, mutate, match):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 10.0}, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        fisher.load_fisher_recordings(path)


def make_fisher_corpus(tmp_path, recs: dict[str, float]):
    """Corpus root with mono source flacs in the documented sidon layout."""
    root = tmp_path / "fisher"
    write_fisher_recordings(root / "m" / "recordings.jsonl.gz", recs)
    loaded = fisher.load_fisher_recordings(root / "m" / "recordings.jsonl.gz")
    for rec in loaded.values():
        for rel in fisher.channel_source_relpaths(rec):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            write_flac(path, num_channels=1, duration_s=rec.duration, sr=24000)
    return root, loaded


def _fake_ffmpeg_merge(monkeypatch, fail_for=None, wrong_frames_for=None):
    """Replace subprocess.run: 'merges' by writing a stereo flac to the last
    argv element, sized from the -i inputs' recording id."""

    def run(cmd, check):
        out = Path(cmd[-1])
        src_a = Path(cmd[cmd.index("-i") + 1])
        rid = src_a.stem[:-2]  # strip "-A"
        if fail_for and rid in fail_for:
            raise subprocess.CalledProcessError(1, cmd)
        import soundfile as sf

        frames = sf.info(str(src_a)).frames
        if wrong_frames_for and rid in wrong_frames_for:
            frames //= 2
        write_flac(out, num_channels=2, duration_s=frames / 24000, sr=24000)
        return None

    monkeypatch.setattr(fisher.subprocess, "run", run)


def test_merge_all_writes_skips_and_is_atomic(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    root, recs = make_fisher_corpus(
        tmp_path, {"fe_03_00001": 8.0, "fe_03_00002": 8.0}
    )
    flac_dir = tmp_path / "flac"
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 2
    assert (flac_dir / "000/fe_03_00001.flac").is_file()
    assert not list(flac_dir.rglob("*.tmp"))
    # idempotent: nothing rewritten on the second pass
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 0


def test_merge_all_removes_stale_tmp_files(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    flac_dir = tmp_path / "flac"
    (flac_dir / "000").mkdir(parents=True)
    stale = flac_dir / "000/fe_03_00001.flac.12345.tmp"
    stale.write_bytes(b"garbage from a killed prior run")
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 1
    assert not stale.exists()


def test_merge_failure_leaves_no_final_file(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch, fail_for={"fe_03_00001"})
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    with pytest.raises(Exception):
        fisher.merge_all(recs, root, tmp_path / "flac", workers=1)
    assert not (tmp_path / "flac/000/fe_03_00001.flac").exists()


def test_merge_frame_mismatch_raises_and_publishes_nothing(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch, wrong_frames_for={"fe_03_00001"})
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    with pytest.raises(RuntimeError, match="frames"):
        fisher.merge_all(recs, root, tmp_path / "flac", workers=1)
    assert not (tmp_path / "flac/000/fe_03_00001.flac").exists()


def test_merge_missing_source_raises(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 8.0})
    recs = fisher.load_fisher_recordings(path)
    with pytest.raises(FileNotFoundError):
        fisher.merge_all(recs, tmp_path / "empty-corpus", tmp_path / "flac", workers=1)
