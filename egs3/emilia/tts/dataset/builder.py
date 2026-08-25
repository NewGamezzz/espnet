"""Emilia dataset builder for the ESPnet3 F5-TTS recipe.

Emits one merged TSV per split over EN+ZH, plus a shard table that maps a
row's ``shard_idx`` back to its directory. The indirection is required:
Emilia's tars split each logical batch into ten sub-shards, so ``EN-B000121``
holds ``EN_B00012_*`` utterances and the sub-shard index appears nowhere in
the utterance id.
"""

from __future__ import annotations

import json
import logging
import os
import time
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Import the submodule directly, NOT `from ...dataset import filters`.
# dataset/__init__.py imports this builder, so a package-level import would
# resolve against a partially-initialized package.
from egs3.emilia.tts.dataset.filters import keep_utterance, normalize_text
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# Bump whenever a change to dataset/filters.py alters which rows a shard
# yields, or what text is stored. It is part of the shard-cache fingerprint,
# so bumping it invalidates every cached shard instead of silently mixing rows
# produced under different filter semantics.
#
# 2: repetition_found corrected to upstream's character n-grams with
#    tolerance=10 (was word n-grams rejecting on a second occurrence), and
#    normalize_text/keep_utterance stopped stripping, so the leading space
#    Emilia ships on every EN record is preserved as upstream does.
_FILTER_VERSION = 2

_HIST_BINS = 50


def _load_cfg(recipe_dir: Path) -> dict:
    cfg_path = Path(recipe_dir) / "dataset" / "config.yaml"
    return load_config_with_defaults(str(cfg_path), resolve=False)["builder"]


def _scan_shard(args) -> tuple[int, Counter]:
    """Read one shard directory and checkpoint it. Runs in a worker process.

    Returns ``(n_rows, dropped)`` rather than the rows themselves: the rows go
    to a per-shard cache file, so they never cross the process boundary and the
    parent never has to hold the whole corpus in memory.
    """
    (shard_rel, shard_idx, lang, corpus_root, lo, hi, strict, audio_suffix,
     cache_dir) = args
    shard_dir = Path(corpus_root) / "emilia" / shard_rel
    rows: list = []
    dropped: Counter = Counter()
    try:
        names = os.listdir(shard_dir)
    except OSError:
        dropped["unreadable_shard"] += 1
        names = []

    audios = {n for n in names if n.endswith(audio_suffix)}
    for name in names:
        if not name.endswith(".json"):
            continue
        if name[:-5] + audio_suffix not in audios:
            dropped["missing_audio"] += 1
            continue
        try:
            with open(shard_dir / name, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            dropped["unreadable_json"] += 1
            continue

        keep, reason = keep_utterance(record, lang, lo, hi, strict)
        if not keep:
            dropped[reason] += 1
            continue
        rows.append(
            (
                record["id"],
                shard_idx,
                lang,
                float(record["duration"]),
                normalize_text(record["text"], lang),
            )
        )
    # Sort within the shard so the build is order-independent across workers.
    rows.sort(key=lambda r: r[0])

    # Checkpoint this shard. The full build is ~2,060 shards and 15-20 hours;
    # without this a walltime kill loses everything, because the parent used to
    # hold all ~38.1M rows in memory and write nothing until the very end.
    #
    # The .json is written LAST and both writes are atomic (temp + os.replace),
    # so the presence of the .json is the completion marker: a job killed
    # mid-write leaves either no .json or a stale temp file, never a truncated
    # pair that a resumed run would mistake for finished work.
    tsv_path = Path(cache_dir) / f"{shard_idx:06d}.tsv"
    meta_path = Path(cache_dir) / f"{shard_idx:06d}.json"
    tmp = tsv_path.with_suffix(".tsv.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for utt_id, idx, lg, duration, text in rows:
            fh.write(f"{utt_id}\t{idx}\t{lg}\t{duration!r}\t{text}\n")
    os.replace(tmp, tsv_path)

    tmp = meta_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"n_rows": len(rows), "dropped": dict(dropped)}, fh)
    os.replace(tmp, meta_path)

    return len(rows), dropped


class EmiliaBuilder(DatasetBuilder):
    """Build Emilia EN+ZH manifests from the staged corpus."""

    def _paths(self, recipe_dir):
        recipe_root = Path(recipe_dir).resolve()
        cfg = _load_cfg(recipe_root)
        data_dir = recipe_root / cfg["data_path"]
        return recipe_root, cfg, data_dir

    def is_source_prepared(self, recipe_dir, **_kwargs) -> bool:
        _, cfg, _ = self._paths(recipe_dir)
        root = Path(cfg["corpus_root"]) / "emilia"
        return all((root / lang).is_dir() for lang in cfg["langs"])

    def prepare_source(self, recipe_dir, **_kwargs) -> None:
        _, cfg, _ = self._paths(recipe_dir)
        root = Path(cfg["corpus_root"]) / "emilia"
        missing = [
            str(root / lang) for lang in cfg["langs"] if not (root / lang).is_dir()
        ]
        raise RuntimeError(
            "The Emilia corpus is staged and read-only; this recipe never "
            "downloads it. Point builder.corpus_root at the staged tree "
            "(default /ocean/projects/cis210027p/ttrachu/emilia_dataset/raw)."
            "\nMissing:\n" + "\n".join(f"  {p}" for p in missing)
        )

    def is_built(self, recipe_dir, **_kwargs) -> bool:
        _, cfg, data_dir = self._paths(recipe_dir)
        expected = list(cfg["manifest_paths"].values())
        expected.append(cfg["shard_table_path"])
        return all((data_dir / rel).is_file() for rel in expected)

    def build(self, recipe_dir, **_kwargs) -> None:
        recipe_root, cfg, data_dir = self._paths(recipe_dir)
        corpus_root = Path(cfg["corpus_root"])
        lo = float(cfg["min_duration"])
        hi = float(cfg["max_duration"])
        strict = bool(cfg["strict_text_filters"])
        # str() so the value stays picklable when EMILIA_BUILD_WORKERS > 1
        # sends this tuple to a ProcessPoolExecutor worker (matches the
        # existing str(corpus_root) below, for the same reason).
        audio_suffix = str(cfg.get("audio_suffix", ".mp3"))

        # Deterministic shard ordering: language order from config, then
        # directory name. shard_idx is the line number in shards.txt.
        shard_rels: list[tuple[str, str]] = []
        for lang in cfg["langs"]:
            lang_dir = corpus_root / "emilia" / lang
            if not lang_dir.is_dir():
                raise FileNotFoundError(f"Missing language dir: {lang_dir}")
            for d in sorted(p.name for p in lang_dir.iterdir() if p.is_dir()):
                shard_rels.append((f"{lang}/{d}", lang))

        manifest_dir = data_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = manifest_dir / ".shard_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # A stale cache is worse than no cache: it would silently mix rows
        # produced by different filter settings. Fingerprint everything that
        # changes which rows a shard yields, and refuse to reuse a cache built
        # under different terms rather than guessing.
        fingerprint = {
            "corpus_root": str(corpus_root),
            "langs": list(cfg["langs"]),
            "min_duration": lo,
            "max_duration": hi,
            "strict_text_filters": strict,
            "audio_suffix": audio_suffix,
            "filter_version": _FILTER_VERSION,
        }
        fp_path = cache_dir / "fingerprint.json"
        if fp_path.is_file():
            previous = json.loads(fp_path.read_text(encoding="utf-8"))
            if previous != fingerprint:
                raise RuntimeError(
                    "Shard cache at %s was built with different settings and "
                    "cannot be reused.\n  cached: %s\n  current: %s\n"
                    "Delete the directory to rebuild from scratch."
                    % (cache_dir, previous, fingerprint)
                )
        else:
            fp_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

        all_jobs = [
            (rel, idx, lang, str(corpus_root), lo, hi, strict, audio_suffix,
             str(cache_dir))
            for idx, (rel, lang) in enumerate(shard_rels)
        ]
        # The .json is written last by the worker, so it marks a complete pair.
        done_meta: dict[int, dict] = {}
        jobs = []
        for job in all_jobs:
            idx = job[1]
            meta_path = cache_dir / f"{idx:06d}.json"
            tsv_path = cache_dir / f"{idx:06d}.tsv"
            if meta_path.is_file() and tsv_path.is_file():
                try:
                    done_meta[idx] = json.loads(meta_path.read_text(encoding="utf-8"))
                    continue
                except (OSError, json.JSONDecodeError):
                    pass  # unreadable -> rescan it
            jobs.append(job)
        if done_meta:
            logger.info(
                "create_dataset: resuming, %d/%d shards already cached in %s",
                len(done_meta), len(all_jobs), cache_dir,
            )

        dropped: Counter = Counter()
        n_rows_total = 0
        for meta in done_meta.values():
            n_rows_total += int(meta["n_rows"])
            dropped.update(meta.get("dropped", {}))
        n_workers = int(os.environ.get("EMILIA_BUILD_WORKERS", "1"))
        # Progress logging is not cosmetic here: the full corpus is ~2,060
        # shards and ~38.1M utterances, the parent accumulates every row in
        # memory, and nothing is written until the loop below finishes. A run
        # that dies on walltime loses all of it, so an operator needs to see
        # the rate early enough to resize the job rather than discover at hour
        # 47 that it was never going to finish.
        t_start = time.time()
        n_jobs = len(jobs)
        log_every = max(1, n_jobs // 100)

        def _accumulate(iterator):
            nonlocal n_rows_total
            scanned = 0
            for done, (shard_n, shard_dropped) in enumerate(iterator, 1):
                n_rows_total += shard_n
                scanned += shard_n
                dropped.update(shard_dropped)
                if done % log_every == 0 or done == n_jobs:
                    elapsed = time.time() - t_start
                    rate = scanned / elapsed if elapsed else 0.0
                    eta = (n_jobs - done) * (elapsed / done) if done else 0.0
                    logger.info(
                        "create_dataset: %d/%d shards this run | %d rows kept "
                        "total | %.0f utt/s | elapsed %.1f h | ETA %.1f h",
                        done, n_jobs, n_rows_total, rate,
                        elapsed / 3600.0, eta / 3600.0,
                    )

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                _accumulate(ex.map(_scan_shard, jobs))
        else:
            _accumulate(_scan_shard(job) for job in jobs)

        # Cache files are read back in shard-index order, which is the same
        # order ex.map produced, so the assembly below is deterministic whether
        # this was one run or a resumed chain of them.
        rng = random.Random(int(cfg["seed"]))
        order = list(range(n_rows_total))
        rng.shuffle(order)
        n_total = len(order)
        if n_total > 1:
            n_val = min(max(1, int(n_total * float(cfg["val_ratio"]))), n_total - 1)
        else:
            n_val = 0
        val_idx = set(order[:n_val])

        train_path = data_dir / cfg["manifest_paths"]["train"]
        valid_path = data_dir / cfg["manifest_paths"]["valid"]
        # Stream the cached shards straight through to the split, rather than
        # materialising ~38.1M rows (~19 GB) to iterate once. Also collects the
        # duration stats in the same pass, so the corpus is read exactly once.
        total_seconds = 0.0
        hist = [0] * _HIST_BINS
        width = (hi - lo) / _HIST_BINS
        i = 0
        with (
            train_path.open("w", encoding="utf-8") as ftr,
            valid_path.open("w", encoding="utf-8") as fva,
        ):
            for idx in range(len(shard_rels)):
                shard_tsv = cache_dir / f"{idx:06d}.tsv"
                if not shard_tsv.is_file():
                    continue
                with shard_tsv.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        (fva if i in val_idx else ftr).write(line)
                        duration = float(line.split("\t", 4)[3])
                        total_seconds += duration
                        b = min(int((duration - lo) / width), _HIST_BINS - 1)
                        hist[max(b, 0)] += 1
                        i += 1
        if i != n_rows_total:
            raise RuntimeError(
                f"shard cache holds {i} rows but the scan counted "
                f"{n_rows_total}; the cache in {cache_dir} is inconsistent"
            )

        (data_dir / cfg["shard_table_path"]).write_text(
            "\n".join(rel for rel, _ in shard_rels) + "\n", encoding="utf-8"
        )

        (manifest_dir / "report.json").write_text(
            json.dumps(
                {
                    "kept": n_rows_total,
                    "dropped": dict(dropped),
                    "total_hours": total_seconds / 3600.0,
                    "duration_histogram": hist,
                    "histogram_range": [lo, hi],
                    "n_shards": len(shard_rels),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "EmiliaBuilder: kept %d, dropped %s, %.1f hours",
            n_rows_total,
            dict(dropped),
            total_seconds / 3600.0,
        )
