"""Stream LEMAS shard tars once and write the manifest's members as FLAC.

Each ``train/<lang>/<shard>.tar.gz`` is read in tar stream mode (gzip is not
seekable), members listed in the shard's poc3k tsv are decoded with soundfile
and written as 16 kHz mono 16-bit FLAC under ``<out_root>/<shard>/``. A
per-shard ``.complete`` marker makes re-runs free and a ``.coverage.json``
records manifest rows versus members found. A member absent from its tar is a
hard failure, which is the audit the mirror runbook deferred to this pass.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Set

import soundfile as sf

logger = logging.getLogger(__name__)


def read_shard_members(tsv_path) -> Set[str]:
    """Return the tar member paths (column 2) listed in a poc3k shard tsv."""
    members = set()
    with Path(tsv_path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                members.add(parts[1])
    return members


def _write_flac(data: bytes, out_path: Path, sample_rate: int) -> None:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        raise RuntimeError(f"{out_path}: expected {sample_rate} Hz, got {sr}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    sf.write(tmp, wav, sample_rate, format="FLAC", subtype="PCM_16")
    tmp.replace(out_path)


def extract_shard(
    tar_path, members: Set[str], out_root, source_sample_rate: int = 16000
) -> dict:
    """Extract ``members`` of one shard tar to FLAC under ``out_root``.

    Args:
        tar_path: ``<shard>.tar.gz``.
        members: Tar member paths to extract (``<shard>/<file>.mp3``).
        out_root: Output directory; files land at
            ``<out_root>/<member with .flac suffix>``.
        source_sample_rate: Expected sample rate of every member.

    Returns:
        Coverage dict ``{"manifest_rows", "members_extracted", "missing"}``.

    Raises:
        RuntimeError: If any member is absent from the tar (the
            ``.complete`` marker is then NOT written), or on a sample-rate
            mismatch.

    Example:
        >>> extract_shard("de000.tar.gz", {"de000/x.mp3"}, "flac/de")
        {'manifest_rows': 1, 'members_extracted': 1, 'missing': []}

    Note:
        A shard whose ``.complete`` marker exists returns its stored coverage
        without touching the tar, so re-running the stage is free.
    """
    tar_path, out_root = Path(tar_path), Path(out_root)
    shard = tar_path.name.split(".")[0]
    done_marker = out_root / f"{shard}.complete"
    coverage_path = out_root / f"{shard}.coverage.json"
    if done_marker.is_file():
        return json.loads(coverage_path.read_text())
    remaining = set(members)
    with tarfile.open(tar_path, "r|gz") as tf:
        for info in tf:
            if info.name not in remaining:
                continue
            data = tf.extractfile(info).read()
            _write_flac(
                data,
                out_root / Path(info.name).with_suffix(".flac"),
                source_sample_rate,
            )
            remaining.discard(info.name)
    coverage = {
        "manifest_rows": len(members),
        "members_extracted": len(members) - len(remaining),
        "missing": sorted(remaining),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage, indent=1))
    if remaining:
        raise RuntimeError(
            f"{tar_path}: {len(remaining)} manifest members absent, "
            f"e.g. {sorted(remaining)[:3]}"
        )
    done_marker.touch()
    return coverage


def _job(args):
    tar_path, tsv_path, out_root, sr = args
    return str(tar_path), extract_shard(
        tar_path, read_shard_members(tsv_path), out_root, sr
    )


def extract_all(
    mirror_root,
    manifest_dir,
    langs: Iterable[str],
    out_root,
    n_workers: int = 32,
    source_sample_rate: int = 16000,
) -> dict:
    """Extract every shard listed under ``<mirror_root>/<manifest_dir>/<lang>``.

    Args:
        mirror_root: LEMAS mirror root (holds ``LEMAS-train/train/<lang>``).
        manifest_dir: poc3k manifest dir, relative to ``mirror_root``.
        langs: Languages to process.
        out_root: FLAC root; files land at ``<out_root>/<lang>/<shard>/``.
        n_workers: Process pool size (one shard per process).
        source_sample_rate: Expected sample rate.

    Returns:
        ``{tar path: coverage dict}``.

    Example:
        >>> extract_all(mirror, "manifests_poc3k", ["de"], flac_root, 8)
    """
    mirror_root, out_root = Path(mirror_root), Path(out_root)
    jobs = []
    for lang in langs:
        for tsv in sorted((mirror_root / manifest_dir / lang).glob("*.tsv")):
            shard = tsv.stem
            tar = mirror_root / "LEMAS-train" / "train" / lang / f"{shard}.tar.gz"
            jobs.append((tar, tsv, out_root / lang, source_sample_rate))
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for tar, cov in pool.map(_job, jobs):
            results[tar] = cov
            logger.info("extracted %s: %s", tar, cov)
    return results
