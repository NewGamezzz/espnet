"""CANDOR manifest parsing and mp3->FLAC transcode machinery."""

import gzip
import json
import string
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.candor_builder import CandorBuilder
from egs3.conversational.tts.dataset.dataset import (
    ConversationDataset,
    read_window_manifest,
)
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


def test_transcode_all_removes_stale_tmp_files(tmp_path, monkeypatch):
    """A `.tmp` left behind by a killed prior run is always abandoned garbage
    (a live run writes under a fresh PID-unique name), so transcode_all
    clears the flac_dir of them before starting new jobs."""
    _fake_ffmpeg(monkeypatch)
    root = make_corpus(tmp_path, ["s1"])
    write_candor_manifests(tmp_path / "m", {"s1": (30.0, [])})
    recs = candor.load_candor_recordings(tmp_path / "m" / "candor_recordings.jsonl.gz")
    flac_dir = tmp_path / "flac"
    flac_dir.mkdir(parents=True)
    stale = flac_dir / "s1.flac.12345.tmp"
    stale.write_bytes(b"garbage from a killed prior run")
    assert candor.transcode_all(recs, root, flac_dir, workers=1) == 1
    assert not stale.exists()
    assert (flac_dir / "s1.flac").is_file()


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


def fabricate_recipe(recipe_dir: Path) -> None:
    tokens = [" "] + list(string.ascii_lowercase) + [".", ","] + ["<turn>", "<OTHER>"]
    vocab = recipe_dir / "data/tokens/vocab.txt"
    vocab.parent.mkdir(parents=True, exist_ok=True)
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def two_speaker_session(duration: float) -> list[dict]:
    """Alternating turns on channels 0/1 covering most of the session."""
    sups, t, ch = [], 1.0, 0
    while t + 4.0 < duration - 1.0:
        sups.append(
            {
                "start": round(t, 2),
                "duration": 3.0,
                "channel": ch,
                "text": "hello there, how are you doing today",
                "speaker": f"spk_{ch}",
            }
        )
        t += 4.0
        ch = 1 - ch
    return sups


def fabricate_candor(tmp_path, durations: dict[str, float]):
    """Corpus manifests + PRE-TRANSCODED flacs (prepare_source not needed)."""
    root = tmp_path / "Candor"
    flac_dir = tmp_path / "candor_flac"
    sessions = {cid: (dur, two_speaker_session(dur)) for cid, dur in durations.items()}
    write_candor_manifests(root / "Candor_lhotse/manifests", sessions)
    for cid, dur in durations.items():
        write_flac(flac_dir / f"{cid}.flac", num_channels=2, duration_s=dur, sr=48000)
    return root, flac_dir


def test_candor_builder_end_to_end(tmp_path):
    root, flac_dir = fabricate_candor(tmp_path, {f"conv-{i}": 40.0 for i in range(4)})
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    builder = CandorBuilder()
    assert builder.is_source_prepared(dataset_root=root, flac_dir=flac_dir)
    builder.build(recipe_dir=recipe, dataset_root=root, flac_dir=flac_dir, seed=0)
    assert builder.is_built(recipe_dir=recipe)

    records = []
    for split in ("train", "valid", "test"):
        path = recipe / f"data/manifest/candor_{split}.jsonl"
        assert path.is_file()
        try:
            records.extend(read_window_manifest(path))
        except RuntimeError:
            pass  # tiny fixture: a split may legitimately be empty
    assert records
    for r in records:
        assert r.num_channels == 2
        assert r.sample_rate == 48000
        assert r.t1 <= 40.0 + 1e-6
        # normalized text: lowercase charset survives
        assert all(t.text == t.text.lower() for t in r.turns)


def test_candor_builder_reports_speaker_overlap(tmp_path, capsys):
    root, flac_dir = fabricate_candor(tmp_path, {f"conv-{i}": 40.0 for i in range(4)})
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    CandorBuilder().build(
        recipe_dir=recipe, dataset_root=root, flac_dir=flac_dir, seed=0
    )
    out = capsys.readouterr().out
    # All 4 tiny sessions land in train (round(4 * 0.02) == 0 for valid/test),
    # each contributing the fixture's fixed spk_0/spk_1 pair; pin the non-zero
    # train speaker count so the assertion cannot pass with a dead
    # accumulator (an unaccumulated `speakers[split]` would print `train(0)`
    # for every pair, same as the untouched valid/test splits).
    assert "speaker overlap train(2)" in out


def test_candor_builder_drops_out_of_range_turn(tmp_path):
    """A supervision starting past the measured audio end clamps to a
    negative-span turn (end <= start) in load_supervisions; the builder must
    drop it rather than emit it into a window."""
    duration = 40.0
    sups = two_speaker_session(duration) + [
        {
            "start": 1000.0,
            "duration": 5.0,
            "channel": 0,
            "text": "ghost beyond audio end",
            "speaker": "spk_ghost",
        }
    ]
    root = tmp_path / "Candor"
    flac_dir = tmp_path / "candor_flac"
    write_candor_manifests(
        root / "Candor_lhotse/manifests", {"conv-oob": (duration, sups)}
    )
    write_flac(
        flac_dir / "conv-oob.flac", num_channels=2, duration_s=duration, sr=48000
    )
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    CandorBuilder().build(
        recipe_dir=recipe, dataset_root=root, flac_dir=flac_dir, seed=0
    )

    records = []
    for split in ("train", "valid", "test"):
        path = recipe / f"data/manifest/candor_{split}.jsonl"
        try:
            records.extend(read_window_manifest(path))
        except RuntimeError:
            pass  # tiny fixture: a split may legitimately be empty
    assert records  # build still succeeds
    assert not any(t.speaker == "spk_ghost" for r in records for t in r.turns)


def test_candor_builder_uses_measured_duration_not_manifest(tmp_path):
    root, flac_dir = fabricate_candor(tmp_path, {"conv-a": 20.0})
    # corrupt the manifest duration upward: windows must still fit real audio
    man = root / "Candor_lhotse/manifests"
    sessions = {"conv-a": (500.0, two_speaker_session(20.0))}
    write_candor_manifests(man, sessions)
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    CandorBuilder().build(
        recipe_dir=recipe, dataset_root=root, flac_dir=flac_dir, seed=0
    )
    records = []
    for split in ("train", "valid", "test"):
        path = recipe / f"data/manifest/candor_{split}.jsonl"
        try:
            records.extend(read_window_manifest(path))
        except RuntimeError:
            pass
    assert records
    assert all(r.t1 <= 20.0 + 1e-3 for r in records)


def test_candor_builder_requires_vocab(tmp_path):
    root, flac_dir = fabricate_candor(tmp_path, {"conv-a": 40.0})
    with pytest.raises(RuntimeError, match="SSSD build"):
        CandorBuilder().build(
            recipe_dir=tmp_path / "recipe",
            dataset_root=root,
            flac_dir=flac_dir,
        )


def test_candor_manifest_loads_through_conversation_dataset(tmp_path):
    root, flac_dir = fabricate_candor(tmp_path, {f"conv-{i}": 40.0 for i in range(4)})
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    CandorBuilder().build(
        recipe_dir=recipe, dataset_root=root, flac_dir=flac_dir, seed=0
    )
    dataset = ConversationDataset(
        split="train",
        manifest_path=recipe / "data/manifest/candor_train.jsonl",
        dataset_root=flac_dir,
        fs=24000,
    )
    sample = dataset[0]
    assert sample["num_channels"] == 2
    assert sample["speech"].shape[0] == 2  # 48 kHz stereo flac -> resampled rows
