"""CANDOR corpus ingestion: lhotse-manifest parsing and mp3 -> FLAC transcoding.

CANDOR ships 1656 two-party sessions as 48 kHz stereo mp3 (one speaker per
channel) plus lhotse recordings/supervisions manifests.  The supervisions are
field-compatible with the SSSD parser, so ``sssd.load_supervisions`` is
reused verbatim by the builder; this module holds only what is
CANDOR-specific: the recordings loader (which points windows at transcoded
FLACs), the documented ``candor_data/<cid>/processed/<cid>.mp3`` source
layout, and the one-time mp3 -> FLAC transcode.  Transcoding exists because
libsndfile in the training environments has no MPEG support (verified
2026-07-31), and mp3 has no sample-accurate seek index for the per-window
random reads training performs.
"""

from __future__ import annotations

import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from .sssd import Recording, _iter_jsonl_gz


def load_candor_recordings(path: str | Path) -> dict[str, Recording]:
    """Parse ``candor_recordings.jsonl.gz`` into ``Recording`` objects.

    ``audio_relpath`` is the transcoded FLAC name ``<cid>.flac``, relative to
    the FLAC directory (the training entries' ``dataset_root``) - never the
    mp3 source path, whose absolute form is machine-specific and untrusted.
    """
    recordings: dict[str, Recording] = {}
    for rec in _iter_jsonl_gz(Path(path)):
        channel_ids = rec.get("channel_ids") or rec["sources"][0]["channels"]
        recordings[rec["id"]] = Recording(
            id=rec["id"],
            audio_relpath=f"{rec['id']}.flac",
            sample_rate=int(rec["sampling_rate"]),
            num_channels=len(channel_ids),
            duration=float(rec["duration"]),
        )
    return recordings


def mp3_relpath(cid: str) -> str:
    """Source location per the corpus README's documented layout."""
    return f"candor_data/{cid}/processed/{cid}.mp3"


def _transcode_one(job: tuple[str, str, str, str]) -> tuple[str, bool]:
    """(cid, mp3_path, flac_path, ffmpeg) -> (cid, newly_written).

    Atomic: encodes to a PID-unique ``<name>.flac.<pid>.tmp`` then
    ``os.replace``s onto the final path, so a killed run never leaves a
    truncated ``.flac`` that the skip-existing check would treat as done.
    The PID suffix matters because two concurrent transcode runs (e.g. two
    overlapping compute jobs) would otherwise share one tmp path and could
    interleave writes into it, and ``os.replace`` would then publish that
    garbage as a permanently "done" final file.  ``-f flac`` is explicit
    because the ``.tmp`` suffix defeats ffmpeg's extension-based format
    inference.
    """
    cid, mp3_path, flac_path, ffmpeg = job
    mp3, flac = Path(mp3_path), Path(flac_path)
    if flac.is_file():
        return cid, False
    if not mp3.is_file():
        raise FileNotFoundError(f"CANDOR source audio not found: {mp3}")
    tmp = flac.with_name(f"{flac.name}.{os.getpid()}.tmp")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-c:a",
            "flac",
            "-f",
            "flac",
            str(tmp),
        ],
        check=True,
    )
    os.replace(tmp, flac)
    return cid, True


def transcode_all(
    recordings: dict[str, Recording],
    corpus_root: str | Path,
    flac_dir: str | Path,
    ffmpeg: str = "ffmpeg",
    workers: int = 4,
) -> int:
    """Idempotent parallel mp3 -> FLAC for every recording; returns how many
    files were newly written.  ``workers <= 1`` runs serially in-process
    (also what the tests use, since a Pool would not see monkeypatches)."""
    flac_dir = Path(flac_dir)
    flac_dir.mkdir(parents=True, exist_ok=True)
    # Any ``*.tmp`` here is assumed to be garbage abandoned by a killed prior
    # run (this call is not expected to overlap another live transcode run
    # against the same flac_dir); best-effort delete so it never accumulates
    # as dead weight. If two transcode runs against the same flac_dir DO
    # overlap, this can unlink the other run's in-flight tmp, which then
    # fails loudly at its own os.replace -- a hard failure, never a silently
    # published corrupt final file.
    for stale in flac_dir.glob("*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass
    jobs = [
        (
            cid,
            str(Path(corpus_root) / mp3_relpath(cid)),
            str(flac_dir / rec.audio_relpath),
            ffmpeg,
        )
        for cid, rec in sorted(recordings.items())
        if not (flac_dir / rec.audio_relpath).is_file()
    ]
    if not jobs:
        return 0
    written = 0
    if int(workers) <= 1:
        for job in jobs:
            written += int(_transcode_one(job)[1])
        return written
    with Pool(processes=int(workers)) as pool:
        for _cid, did_write in pool.imap_unordered(_transcode_one, jobs):
            written += int(did_write)
    return written


def measured_durations(
    recordings: dict[str, Recording], flac_dir: str | Path
) -> dict[str, float]:
    """Actual decoded duration per session, from the FLAC headers.

    The manifests' durations describe the mp3s; mp3 encoder delay/padding can
    shift decoded lengths, and windows must never overrun the real audio, so
    the builder replaces every duration with this measurement.  Also verifies
    rate and channel count so a wrong or stale transcode fails loudly here
    rather than mid-training.
    """
    import soundfile as sf

    durations: dict[str, float] = {}
    for cid, rec in sorted(recordings.items()):
        info = sf.info(str(Path(flac_dir) / rec.audio_relpath))
        if info.samplerate != rec.sample_rate:
            raise RuntimeError(
                f"{rec.audio_relpath}: sample rate {info.samplerate} != "
                f"manifest {rec.sample_rate}"
            )
        if info.channels != rec.num_channels:
            raise RuntimeError(
                f"{rec.audio_relpath}: {info.channels} channels != "
                f"manifest {rec.num_channels}"
            )
        durations[cid] = info.frames / info.samplerate
    return durations
