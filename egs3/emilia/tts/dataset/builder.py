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

_HIST_BINS = 50


def _load_cfg(recipe_dir: Path) -> dict:
    cfg_path = Path(recipe_dir) / "dataset" / "config.yaml"
    return load_config_with_defaults(str(cfg_path), resolve=False)["builder"]


def _scan_shard(args) -> tuple[list, Counter]:
    """Read one shard directory. Runs in a worker process."""
    shard_rel, shard_idx, lang, corpus_root, lo, hi, strict, audio_suffix = args
    shard_dir = Path(corpus_root) / "emilia" / shard_rel
    rows: list = []
    dropped: Counter = Counter()
    try:
        names = os.listdir(shard_dir)
    except OSError:
        return rows, dropped

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
        rows.append((
            record["id"], shard_idx, lang,
            float(record["duration"]),
            normalize_text(record["text"], lang),
        ))
    # Sort within the shard so the build is order-independent across workers.
    rows.sort(key=lambda r: r[0])
    return rows, dropped


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
        missing = [str(root / lang) for lang in cfg["langs"]
                   if not (root / lang).is_dir()]
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

        jobs = [
            (rel, idx, lang, str(corpus_root), lo, hi, strict, audio_suffix)
            for idx, (rel, lang) in enumerate(shard_rels)
        ]

        rows: list = []
        dropped: Counter = Counter()
        n_workers = int(os.environ.get("EMILIA_BUILD_WORKERS", "1"))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = ex.map(_scan_shard, jobs)
                for shard_rows, shard_dropped in results:
                    rows.extend(shard_rows)
                    dropped.update(shard_dropped)
        else:
            for job in jobs:
                shard_rows, shard_dropped = _scan_shard(job)
                rows.extend(shard_rows)
                dropped.update(shard_dropped)

        # ex.map preserves input order, so `rows` is already deterministic.
        rng = random.Random(int(cfg["seed"]))
        order = list(range(len(rows)))
        rng.shuffle(order)
        n_total = len(order)
        if n_total > 1:
            n_val = min(
                max(1, int(n_total * float(cfg["val_ratio"]))), n_total - 1
            )
        else:
            n_val = 0
        val_idx = set(order[:n_val])

        manifest_dir = data_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        train_path = data_dir / cfg["manifest_paths"]["train"]
        valid_path = data_dir / cfg["manifest_paths"]["valid"]
        with train_path.open("w", encoding="utf-8") as ftr, \
                valid_path.open("w", encoding="utf-8") as fva:
            for i, (utt_id, shard_idx, lang, duration, text) in enumerate(rows):
                line = f"{utt_id}\t{shard_idx}\t{lang}\t{duration!r}\t{text}\n"
                (fva if i in val_idx else ftr).write(line)

        (data_dir / cfg["shard_table_path"]).write_text(
            "\n".join(rel for rel, _ in shard_rels) + "\n", encoding="utf-8"
        )

        total_seconds = sum(r[3] for r in rows)
        hist = [0] * _HIST_BINS
        width = (hi - lo) / _HIST_BINS
        for r in rows:
            b = min(int((r[3] - lo) / width), _HIST_BINS - 1)
            hist[max(b, 0)] += 1
        (manifest_dir / "report.json").write_text(json.dumps({
            "kept": len(rows),
            "dropped": dict(dropped),
            "total_hours": total_seconds / 3600.0,
            "duration_histogram": hist,
            "histogram_range": [lo, hi],
            "n_shards": len(shard_rels),
        }, indent=2), encoding="utf-8")

        logger.info(
            "EmiliaBuilder: kept %d, dropped %s, %.1f hours",
            len(rows), dict(dropped), total_seconds / 3600.0,
        )
