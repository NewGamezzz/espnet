"""CANDOR manifest parsing and mp3->FLAC transcode machinery."""

import gzip
import json
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.preprocessing import candor

from .conftest import REPO_ROOT, write_flac  # noqa: F401  (sys.path setup)


def write_candor_manifests(
    manifest_dir: Path, sessions: dict[str, tuple[float, list[dict]]]
) -> None:
    """sessions: cid -> (duration_s, [supervision dicts]). Writes the two
    jsonl.gz manifests in the real corpus layout/fields."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(manifest_dir / "candor_recordings.jsonl.gz", "wt") as f:
        for cid, (duration, _sups) in sorted(sessions.items()):
            f.write(
                json.dumps(
                    {
                        "id": cid,
                        "sources": [
                            {
                                "type": "file",
                                "channels": [0, 1],
                                "source": f"/somewhere/else/{cid}.mp3",
                            }
                        ],
                        "sampling_rate": 48000,
                        "num_samples": round(duration * 48000),
                        "duration": duration,
                        "channel_ids": [0, 1],
                    }
                )
                + "\n"
            )
    with gzip.open(
        manifest_dir / "candor_supervisions_cliffhanger.jsonl.gz", "wt"
    ) as f:
        for cid, (_duration, sups) in sorted(sessions.items()):
            for i, sup in enumerate(sups):
                f.write(
                    json.dumps(
                        {
                            "id": f"{cid}-cliffhanger-{i:05d}-{sup['channel']}",
                            "recording_id": cid,
                            "language": "English",
                            **sup,
                        }
                    )
                    + "\n"
                )


def test_load_candor_recordings_flac_relpath(tmp_path):
    write_candor_manifests(tmp_path, {"abc-123": (30.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "candor_recordings.jsonl.gz")
    rec = recs["abc-123"]
    # points at the TRANSCODED flac, never the mp3 source path
    assert rec.audio_relpath == "abc-123.flac"
    assert (rec.sample_rate, rec.num_channels) == (48000, 2)
    assert rec.duration == 30.0


def test_mp3_relpath_documented_layout():
    assert candor.mp3_relpath("abc-123") == "candor_data/abc-123/processed/abc-123.mp3"


def _fake_ffmpeg(monkeypatch, fail_for: set[str] | None = None):
    """Replace subprocess.run: 'transcodes' by writing a tiny flac to the
    output path (last argv element). Raises for cids in fail_for."""
    import subprocess

    def run(cmd, check):
        out = Path(cmd[-1])
        src = Path(cmd[cmd.index("-i") + 1])
        if fail_for and any(cid in src.name for cid in fail_for):
            raise subprocess.CalledProcessError(1, cmd)
        write_flac(out, num_channels=2, duration_s=1.0, sr=48000)
        return None

    monkeypatch.setattr(candor.subprocess, "run", run)


def make_corpus(tmp_path, cids):
    root = tmp_path / "corpus"
    for cid in cids:
        mp3 = root / candor.mp3_relpath(cid)
        mp3.parent.mkdir(parents=True, exist_ok=True)
        mp3.write_bytes(b"\x00")  # placeholder; fake ffmpeg never reads it
    return root


def test_transcode_all_writes_skips_and_is_atomic(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch)
    root = make_corpus(tmp_path, ["s1", "s2"])
    write_candor_manifests(tmp_path / "m", {"s1": (30.0, []), "s2": (30.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    flac_dir = tmp_path / "flac"
    # workers=1 runs serially in-process so the monkeypatch applies
    assert candor.transcode_all(recs, root, flac_dir, workers=1) == 2
    assert (flac_dir / "s1.flac").is_file() and (flac_dir / "s2.flac").is_file()
    assert not list(flac_dir.glob("*.tmp"))
    # idempotent: nothing rewritten on the second pass
    assert candor.transcode_all(recs, root, flac_dir, workers=1) == 0


def test_transcode_failure_leaves_no_final_file(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, fail_for={"s1"})
    root = make_corpus(tmp_path, ["s1"])
    write_candor_manifests(tmp_path / "m", {"s1": (30.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    flac_dir = tmp_path / "flac"
    with pytest.raises(Exception):
        candor.transcode_all(recs, root, flac_dir, workers=1)
    assert not (flac_dir / "s1.flac").exists()


def test_transcode_missing_mp3_raises(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch)
    write_candor_manifests(tmp_path / "m", {"s1": (30.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    with pytest.raises(FileNotFoundError):
        candor.transcode_all(
            recs, tmp_path / "empty-corpus", tmp_path / "flac", workers=1
        )


def test_measured_durations_reads_flac_headers(tmp_path):
    write_candor_manifests(tmp_path / "m", {"s1": (999.0, [])})  # manifest lies
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    flac_dir = tmp_path / "flac"
    write_flac(flac_dir / "s1.flac", num_channels=2, duration_s=8.0, sr=48000)
    durs = candor.measured_durations(recs, flac_dir)
    assert abs(durs["s1"] - 8.0) < 1e-3  # measured, not the manifest's 999


def test_measured_durations_rejects_wrong_rate_or_channels(tmp_path):
    write_candor_manifests(tmp_path / "m", {"s1": (8.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    flac_dir = tmp_path / "flac"
    write_flac(flac_dir / "s1.flac", num_channels=1, duration_s=8.0, sr=48000)
    with pytest.raises(RuntimeError):
        candor.measured_durations(recs, flac_dir)
