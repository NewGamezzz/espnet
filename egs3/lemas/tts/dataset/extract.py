"""Stream LEMAS shard tars once and pack the manifest's members as raw PCM.

Each ``train/<lang>/<shard>.tar.gz`` is read in tar stream mode (gzip is not
seekable); members listed in the shard's poc3k tsv are decoded with soundfile,
downmixed, resampled with soxr when they are not at 16 kHz (the Emilia portions
of en/zh ship at 24/32 kHz; counted in the coverage) and appended in tar order
to ONE file per shard::

    <out_root>/<shard>.pcm        int16 little-endian mono 16 kHz, concatenated
    <out_root>/<shard>.index.tsv  <member> <start_sample> <n_samples>

One file per shard rather than one per member because random access to 30 M
small files on Lustre costs ~160 ms per open (metadata) while a ``pread`` of a
region inside a large file costs ~40 ms (measured on Delta /work/hdd), and the
dataset reads three regions per training item. A per-shard ``.complete``
marker makes re-runs free and ``.coverage.json`` records manifest rows versus
members found. A member absent from its tar is a hard failure, which is the
audit the mirror runbook deferred to this pass.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger(__name__)

PCM_DTYPE = "<i2"  # int16 little-endian, 2 bytes per sample


def read_shard_members(tsv_path) -> Set[str]:
    """Return the tar member paths (column 2) listed in a poc3k shard tsv."""
    members = set()
    with Path(tsv_path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                members.add(parts[1])
    return members


def read_pack_index(index_path) -> Dict[str, Tuple[int, int]]:
    """Return ``member -> (start_sample, n_samples)`` from a ``.index.tsv``.

    Args:
        index_path: ``<shard>.index.tsv`` written by :func:`extract_shard`.

    Returns:
        Mapping from tar member path to its region in ``<shard>.pcm``.

    Example:
        >>> read_pack_index("pcm/de/de000.index.tsv")["de000/x.mp3"]
        (0, 38400)
    """
    index: Dict[str, Tuple[int, int]] = {}
    with Path(index_path).open(encoding="utf-8") as f:
        for line in f:
            member, start, n = line.rstrip("\n").split("\t")
            index[member] = (int(start), int(n))
    return index


def _decode(data: bytes, sample_rate: int) -> Tuple[np.ndarray, bool]:
    """Decode, downmix, resample to ``sample_rate`` if needed, quantise to int16.

    Returns the int16 samples and whether the member had to be resampled.
    """
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    resampled = sr != sample_rate
    if resampled:
        wav = soxr.resample(wav, sr, sample_rate, quality="HQ")
    pcm = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(PCM_DTYPE)
    return pcm, resampled


def extract_shard(
    tar_path, members: Set[str], out_root, source_sample_rate: int = 16000
) -> dict:
    """Pack ``members`` of one shard tar into ``<out_root>/<shard>.pcm``.

    Args:
        tar_path: ``<shard>.tar.gz``.
        members: Tar member paths to pack (``<shard>/<file>.mp3``).
        out_root: Output directory for ``<shard>.pcm`` and ``<shard>.index.tsv``.
        source_sample_rate: Output sample rate; members at another rate are
            resampled to it.

    Returns:
        Coverage dict ``{"manifest_rows", "members_extracted", "resampled",
        "samples", "missing"}``.

    Raises:
        RuntimeError: If any member is absent from the tar (the
            ``.complete`` marker is then NOT written).

    Example:
        >>> extract_shard("de000.tar.gz", {"de000/x.mp3"}, "pcm/de")
        {'manifest_rows': 1, 'members_extracted': 1, 'resampled': 0, ...}

    Note:
        A shard whose ``.complete`` marker exists returns its stored coverage
        without touching the tar, so re-running the stage is free. A partial
        ``.pcm.tmp`` from an interrupted run is overwritten: a gzip stream
        cannot be resumed mid-way.
    """
    tar_path, out_root = Path(tar_path), Path(out_root)
    shard = tar_path.name.split(".")[0]
    done_marker = out_root / f"{shard}.complete"
    coverage_path = out_root / f"{shard}.coverage.json"
    if done_marker.is_file():
        return json.loads(coverage_path.read_text())
    out_root.mkdir(parents=True, exist_ok=True)
    pack = out_root / f"{shard}.pcm"
    tmp = out_root / f"{shard}.pcm.tmp"
    remaining = set(members)
    n_resampled = pos = 0
    index = []
    with tarfile.open(tar_path, "r|gz") as tf, tmp.open("wb") as fpcm:
        for info in tf:
            if info.name not in remaining:
                continue
            pcm, resampled = _decode(tf.extractfile(info).read(), source_sample_rate)
            fpcm.write(pcm.tobytes())
            index.append((info.name, pos, len(pcm)))
            pos += len(pcm)
            n_resampled += resampled
            remaining.discard(info.name)
    (out_root / f"{shard}.index.tsv").write_text(
        "".join(f"{m}\t{s}\t{n}\n" for m, s, n in index), encoding="utf-8"
    )
    tmp.replace(pack)
    coverage = {
        "manifest_rows": len(members),
        "members_extracted": len(members) - len(remaining),
        "resampled": n_resampled,
        "samples": pos,
        "missing": sorted(remaining),
    }
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
    """Pack every shard listed under ``<mirror_root>/<manifest_dir>/<lang>``.

    Args:
        mirror_root: LEMAS mirror root (holds ``LEMAS-train/train/<lang>``).
        manifest_dir: poc3k manifest dir, relative to ``mirror_root``.
        langs: Languages to process.
        out_root: Pack root; files land at ``<out_root>/<lang>/<shard>.pcm``.
        n_workers: Process pool size (one shard per process).
        source_sample_rate: Output sample rate.

    Returns:
        ``{tar path: coverage dict}``.

    Example:
        >>> extract_all(mirror, "manifests_poc3k", ["de"], pcm_root, 8)
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
