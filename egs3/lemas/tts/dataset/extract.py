"""Stream LEMAS shard tars once and pack the manifest's members as raw PCM.

Each ``train/<lang>/<shard>.tar.gz`` is read in tar stream mode (gzip is not
seekable); members listed in the shard's poc3k tsv are decoded with soundfile,
downmixed, brought to the model rate (24 kHz) with soxr and appended in tar
order to ONE file per shard. LEMAS is 16 kHz except the Emilia portions of
en/zh (24/32 kHz): 16 kHz rows are upsampled once here instead of per item in
the loader, 24 kHz rows are stored as they are, 32 kHz rows come down to the
model's Nyquist; nothing above 8 kHz is invented for the 16 kHz sources and
nothing the model could use is lost for Emilia. ``resampled`` in the coverage
counts members whose rate was not 24 kHz::

    <out_root>/<shard>.pcm        int16 little-endian mono 24 kHz, concatenated
    <out_root>/<shard>.index.tsv  <member> <start_sample> <n_samples>

One file per shard rather than one per member because random access to 30 M
small files on Lustre costs ~160 ms per open (metadata) while a ``pread`` of a
region inside a large file costs ~40 ms (measured on Delta /work/hdd), and the
dataset reads three regions per training item. A per-shard ``.complete``
marker makes re-runs free and ``.coverage.json`` records manifest rows versus
members found. A member absent from its tar is a hard failure, which is the
audit the mirror runbook deferred to this pass.

Second pass (:func:`regroup_language`): the poc3k rows were subsampled, so a
speaker's segments are scattered over a shard pack (0.8% of grouped rows have
a same-group row within 4 MB) and random 64 KB preads cost 60-75 ms with a
p90 of 200-360 ms under load, while a 4 MB aligned read costs ~100 ms. The
shard packs are therefore re-laid out into ONE pack per language in which
rows are grouped into chunks of at most ``chunk_rows`` rows of one speaker /
recording (in segment order) and the chunks are shuffled, so that the dataset
can serve a whole batch plus its prompts from one or two 4 MB blocks::

    <out_root>/<lang>/<lang>.pcm        int16 mono 24 kHz, chunk-contiguous
    <out_root>/<lang>/<lang>.index.tsv  <member> <start_sample> <n_samples>
    <out_root>/<lang>/<lang>.regrouped  marker (+ .regroup.json stats)

Both passes are sequential I/O: pass 1 streams each shard pack once and
appends rows to per-shard hash buckets keyed by ``(group, sub-chunk)``; pass 2
reads one bucket at a time, orders it into chunks, shuffles them and appends
to the language pack.
"""

from __future__ import annotations

import io
import json
import logging
import math
import random
import re
import shutil
import tarfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import soxr
from src.layout import PACK_SR

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

    Returns the int16 samples and whether the member's rate was not ``sample_rate``.
    """
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    resampled = sr != sample_rate
    if resampled:
        wav = soxr.resample(wav, sr, sample_rate, quality="HQ")
    pcm = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(PCM_DTYPE)
    return pcm, resampled


def extract_shard(
    tar_path, members: Set[str], out_root, sample_rate: int = PACK_SR
) -> dict:
    """Pack ``members`` of one shard tar into ``<out_root>/<shard>.pcm``.

    Args:
        tar_path: ``<shard>.tar.gz``.
        members: Tar member paths to pack (``<shard>/<file>.mp3``).
        out_root: Output directory for ``<shard>.pcm`` and ``<shard>.index.tsv``.
        sample_rate: Output sample rate; members at another rate are
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
            pcm, resampled = _decode(tf.extractfile(info).read(), sample_rate)
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
    sample_rate: int = PACK_SR,
) -> dict:
    """Pack every shard listed under ``<mirror_root>/<manifest_dir>/<lang>``.

    Args:
        mirror_root: LEMAS mirror root (holds ``LEMAS-train/train/<lang>``).
        manifest_dir: poc3k manifest dir, relative to ``mirror_root``.
        langs: Languages to process.
        out_root: Pack root; files land at ``<out_root>/<lang>/<shard>.pcm``.
        n_workers: Process pool size (one shard per process).
        sample_rate: Output sample rate.

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
            jobs.append((tar, tsv, out_root / lang, sample_rate))
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for tar, cov in pool.map(_job, jobs):
            results[tar] = cov
            logger.info("extracted %s: %s", tar, cov)
    return results


# ---- pass 2: chunk-contiguous per-language packs ------------------------------

CHUNK_ROWS = 32  # rows of one group laid out contiguously (~100 s of audio)
N_BUCKETS = 128  # hash buckets per language; a bucket must fit in RAM (~2.5 GB)
_READ_BYTES = 64 * 1024 * 1024  # sequential read size when streaming a pack


def _crc(*parts: str) -> int:
    return zlib.crc32("\x00".join(parts).encode("utf-8"))


def chunk_key(
    group: str,
    key: str,
    seg: Optional[int],
    group_size: int,
    chunk_rows: int = CHUNK_ROWS,
) -> str:
    """Sub-chunk id for a row: which contiguous run of its group it belongs to.

    Recording groups (segment index known) are cut in segment order; speaker
    groups without a segment index are cut by a stable hash of the key so a
    42k-row speaker spreads over ~1,300 chunks instead of one giant run. Rows
    without a group are their own chunk.
    """
    if not group:
        return key
    n_sub = max(1, math.ceil(group_size / chunk_rows))
    sub = (seg // chunk_rows) if seg is not None else (_crc(key) % n_sub)
    return f"{group}\x01{sub}"


class _SeqReader:
    """Sequential slices out of a pack: rows are requested in increasing order."""

    def __init__(self, path: Path):
        self.f = path.open("rb")
        self.base = 0  # sample index of buf[0]
        self.buf = np.zeros(0, dtype=PCM_DTYPE)

    def read(self, start: int, n: int) -> np.ndarray:
        end = start + n
        if start < self.base:
            raise ValueError("rows must be read in increasing order")
        if end > self.base + len(self.buf):
            keep = self.buf[start - self.base :]
            self.f.seek(start * 2)
            want = max(_READ_BYTES, (end - start) * 2)
            self.buf = np.concatenate(
                [keep, np.frombuffer(self.f.read(want), dtype=PCM_DTYPE)]
            )
            self.base = start
        off = start - self.base
        return self.buf[off : off + n]

    def close(self) -> None:
        self.f.close()


def _bucket_shard(args):
    """Pass 1 for one shard pack: append its rows to per-shard bucket parts."""
    pack, index_path, parts_dir, rows, n_buckets = args
    # rows: member -> (bucket, chunk) computed by the caller from the keys
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    index = read_pack_index(index_path)
    reader = _SeqReader(Path(pack))
    files = {}
    offsets = [0] * n_buckets
    n_rows = 0
    try:
        for member, (start, n) in sorted(index.items(), key=lambda kv: kv[1][0]):
            if member not in rows:
                continue
            bucket, chunk = rows[member]
            fpcm, ftsv = files.get(bucket) or files.setdefault(
                bucket,
                (
                    (parts_dir / f"b{bucket}.pcm").open("wb"),
                    (parts_dir / f"b{bucket}.tsv").open("w", encoding="utf-8"),
                ),
            )
            fpcm.write(reader.read(start, n).tobytes())
            ftsv.write(f"{member}\t{chunk}\t{offsets[bucket]}\t{n}\n")
            offsets[bucket] += n
            n_rows += 1
    finally:
        reader.close()
        for fpcm, ftsv in files.values():
            fpcm.close()
            ftsv.close()
    (parts_dir / ".complete").touch()
    return str(pack), n_rows


def _merge_language(args):
    """Pass 2 for one language: buckets -> chunks -> shuffled language pack."""
    lang, parts_dirs, out_root, n_buckets, seed, chunk_rows = args
    out_root = Path(out_root)
    lang_dir = out_root / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    tmp = lang_dir / f"{lang}.pcm.tmp"
    index_path = lang_dir / f"{lang}.index.tsv"
    rng = random.Random(seed)
    buckets = list(range(n_buckets))
    rng.shuffle(buckets)
    pos = 0
    n_rows = n_chunks = 0
    with tmp.open("wb") as fpcm, index_path.open("w", encoding="utf-8") as fidx:
        for b in buckets:
            chunks: Dict[str, List[Tuple[str, np.ndarray]]] = {}
            for parts_dir in parts_dirs:
                tsv = Path(parts_dir) / f"b{b}.tsv"
                if not tsv.is_file():
                    continue
                pcm = np.fromfile(Path(parts_dir) / f"b{b}.pcm", dtype=PCM_DTYPE)
                with tsv.open(encoding="utf-8") as f:
                    for line in f:
                        member, chunk, start, n = line.rstrip("\n").split("\t")
                        start, n = int(start), int(n)
                        chunks.setdefault(chunk, []).append(
                            (member, pcm[start : start + n])
                        )
            order = sorted(chunks)  # deterministic before the shuffle
            rng.shuffle(order)
            for chunk in order:
                rows = chunks[chunk]
                rows.sort(key=lambda r: _member_sort_key(r[0]))
                for i in range(0, len(rows), chunk_rows):
                    for member, wav in rows[i : i + chunk_rows]:
                        fpcm.write(wav.tobytes())
                        fidx.write(f"{member}\t{pos}\t{len(wav)}\n")
                        pos += len(wav)
                        n_rows += 1
                    n_chunks += 1
    tmp.replace(lang_dir / f"{lang}.pcm")
    stats = {"rows": n_rows, "chunks": n_chunks, "samples": pos}
    (lang_dir / f"{lang}.regroup.json").write_text(json.dumps(stats, indent=1))
    (lang_dir / f"{lang}.regrouped").touch()
    return lang, stats


def _member_sort_key(member: str):
    # segment order inside a chunk: numeric runs in the member name sort by
    # value (yodas "<vid>-00012-..." before "-00013-...")
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", member)]


def regroup_language(
    lang: str,
    shards: List[Tuple[str, str, Dict[str, Tuple[int, str]]]],
    out_root,
    seed: int = 0,
    chunk_rows: int = CHUNK_ROWS,
    n_buckets: int = N_BUCKETS,
    n_workers: int = 8,
    keep_parts: bool = False,
) -> dict:
    """Re-lay out one language's shard packs into a chunk-contiguous pack.

    Args:
        lang: Language code.
        shards: One ``(pack_path, index_path, rows)`` per shard, where
            ``rows`` maps a tar member to ``(bucket, chunk_key)`` (see
            :func:`chunk_key`; the bucket is ``crc32(chunk_key) % n_buckets``).
        out_root: Pack root; writes ``<out_root>/<lang>/<lang>.pcm``.
        seed: Shuffle seed for bucket and chunk order.
        chunk_rows: Maximum rows per contiguous chunk.
        n_buckets: Hash buckets (RAM per bucket ~ pack size / n_buckets).
        n_workers: Processes for pass 1 (one per shard).
        keep_parts: Keep the pass-1 bucket parts (debugging).

    Returns:
        ``{"rows", "chunks", "samples"}`` of the language pack.

    Example:
        >>> regroup_language("de", shards, "pcm", seed=1)["chunks"] > 0
        True

    Note:
        A language whose ``<lang>.regrouped`` marker exists is skipped.
    """
    out_root = Path(out_root)
    lang_dir = out_root / lang
    if (lang_dir / f"{lang}.regrouped").is_file():
        return json.loads((lang_dir / f"{lang}.regroup.json").read_text())
    parts_root = lang_dir / "_parts"
    jobs, parts_dirs = [], []
    for pack, index_path, rows in shards:
        parts_dir = parts_root / Path(pack).stem
        parts_dirs.append(str(parts_dir))
        if (parts_dir / ".complete").is_file():
            continue
        jobs.append((str(pack), str(index_path), str(parts_dir), rows, n_buckets))
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, min(n_workers, len(jobs)))) as pool:
            for pack, n in pool.map(_bucket_shard, jobs):
                logger.info("bucketed %s: %d rows", pack, n)
    _lang, stats = _merge_language(
        (lang, parts_dirs, str(out_root), n_buckets, seed, chunk_rows)
    )
    if not keep_parts:
        shutil.rmtree(parts_root, ignore_errors=True)
    logger.info("regrouped %s: %s", lang, stats)
    return stats


def bucket_of(chunk: str, n_buckets: int = N_BUCKETS) -> int:
    """Hash bucket of a chunk key."""
    return _crc(chunk) % n_buckets


def regroup_all(
    shards_by_lang: Dict[str, List[Tuple[str, str, Dict[str, Tuple[int, str]]]]],
    out_root,
    seed: int = 0,
    chunk_rows: int = CHUNK_ROWS,
    n_buckets: int = N_BUCKETS,
    n_workers: int = 32,
    keep_parts: bool = False,
) -> Dict[str, dict]:
    """Run :func:`regroup_language` for several languages with shared pools.

    Pass 1 runs over all shards of all languages at once (one process per
    shard), pass 2 over the languages at once (one process per language, each
    holding one bucket, ~pack size / ``n_buckets``, in RAM).

    Args:
        shards_by_lang: ``lang -> [(pack_path, index_path, rows)]`` as for
            :func:`regroup_language`.
        out_root: Pack root.
        seed: Shuffle seed; each language uses ``seed + hash(lang)``.
        chunk_rows: Maximum rows per contiguous chunk.
        n_buckets: Hash buckets per language.
        n_workers: Process pool size.
        keep_parts: Keep the pass-1 bucket parts.

    Returns:
        ``lang -> {"rows", "chunks", "samples"}``.

    Example:
        >>> regroup_all({"de": shards}, "pcm", seed=1)["de"]["rows"] > 0
        True
    """
    out_root = Path(out_root)
    stats: Dict[str, dict] = {}
    jobs, merges, parts_roots = [], [], {}
    for lang, shards in shards_by_lang.items():
        lang_dir = out_root / lang
        if (lang_dir / f"{lang}.regrouped").is_file():
            stats[lang] = json.loads((lang_dir / f"{lang}.regroup.json").read_text())
            continue
        parts_root = lang_dir / "_parts"
        parts_roots[lang] = parts_root
        parts_dirs = []
        for pack, index_path, rows in shards:
            parts_dir = parts_root / Path(pack).stem
            parts_dirs.append(str(parts_dir))
            if not (parts_dir / ".complete").is_file():
                jobs.append(
                    (str(pack), str(index_path), str(parts_dir), rows, n_buckets)
                )
        merges.append(
            (lang, parts_dirs, str(out_root), n_buckets, seed + _crc(lang), chunk_rows)
        )
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, min(n_workers, len(jobs)))) as pool:
            for pack, n in pool.map(_bucket_shard, jobs):
                logger.info("bucketed %s: %d rows", pack, n)
    if merges:
        with ProcessPoolExecutor(
            max_workers=max(1, min(n_workers, len(merges)))
        ) as pool:
            for lang, st in pool.map(_merge_language, merges):
                stats[lang] = st
                logger.info("regrouped %s: %s", lang, st)
                if not keep_parts:
                    shutil.rmtree(parts_roots[lang], ignore_errors=True)
    return stats
