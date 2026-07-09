"""SSSD dataset builder: window manifests, splits, and the extended vocab.

Thin orchestration in the libritts house style; the algorithms live in
``sssd.py`` / ``windows.py`` / ``text.py`` so tests hit them with fabricated
fixtures.  The corpus directory is treated as strictly read-only.

Build outputs (under ``<recipe_dir>/data/``):
  - ``manifest/{train,valid,test}.jsonl``  window manifests (one JSON per line)
  - ``tokens/vocab.txt``   base vocab + ``<turn>`` + ``<OTHER>`` appended at the
    end (pure token-per-line; the line index is the token id, so the file
    carries no comments -- ids are documented in the meta file instead)
  - ``tokens/vocab_meta.json``  base path/size and the new token ids
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
from importlib import resources
from pathlib import Path
from typing import Sequence

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from .sssd import (
    Turn,
    load_recordings,
    load_supervisions,
    merge_turns,
    session_speakers,
)
from .text import NEW_TOKENS, extend_vocab, normalize_text, vocab_charset
from .windows import WindowingStats, build_windows, to_json


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def resolve_dataset_root(explicit: str | Path | None = None) -> Path:
    """Corpus root resolution used across the recipe: explicit argument >
    ``$SSSD_ROOT`` > ``builder.dataset_root`` in config.yaml."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("SSSD_ROOT"):
        return Path(os.environ["SSSD_ROOT"])
    return Path(_CFG["dataset_root"])


def split_sessions(
    session_ids: Sequence[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Seeded session-level split; test and valid sizes round first so the
    rounding never eats the small splits, train takes the rest."""
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1, got {total}")
    ids = sorted(session_ids)
    random.Random(f"{seed}:split").shuffle(ids)
    n = len(ids)
    n_test = round(n * ratios["test"])
    n_valid = round(n * ratios["valid"])
    return {
        "test": sorted(ids[:n_test]),
        "valid": sorted(ids[n_test : n_test + n_valid]),
        "train": sorted(ids[n_test + n_valid :]),
    }


def overlap_and_speech_time(turns: Sequence[Turn]) -> tuple[float, float]:
    """(time with >= 2 channels active, time with >= 1 active) via event sweep."""
    events: list[tuple[float, int]] = []
    for t in turns:
        events.append((t.start, 1))
        events.append((t.end, -1))
    events.sort()
    overlap = speech = 0.0
    active = 0
    prev = 0.0
    for time, delta in events:
        if active >= 1:
            speech += time - prev
        if active >= 2:
            overlap += time - prev
        active += delta
        prev = time
    return overlap, speech


def _distribution(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    vals = sorted(values)
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0]] * 3
    return (
        f"n={len(vals)} min={vals[0]:.1f} p25={q[0]:.1f} median={q[1]:.1f} "
        f"p75={q[2]:.1f} max={vals[-1]:.1f} mean={statistics.fmean(vals):.1f}"
    )


class SSSDBuilder(DatasetBuilder):
    """Prepare SSSD window manifests and the extended vocab for ESPnet3 TTS."""

    def is_source_prepared(
        self, recipe_dir: str | Path | None = None, dataset_root: str | Path | None = None, **_
    ) -> bool:
        root = resolve_dataset_root(dataset_root)
        manifests = root / _CFG["manifests_subdir"]
        return (
            (root / _CFG["audio_subdir"]).is_dir()
            and (manifests / "recordings.jsonl.gz").is_file()
            and (manifests / "supervisions.jsonl.gz").is_file()
        )

    def prepare_source(
        self, recipe_dir: str | Path | None = None, dataset_root: str | Path | None = None, **_
    ) -> None:
        if not self.is_source_prepared(dataset_root=dataset_root):
            root = resolve_dataset_root(dataset_root)
            raise RuntimeError(
                f"SSSD corpus not found at {root}. The corpus is externally "
                "provisioned and read-only; point dataset_root (config), "
                "$SSSD_ROOT, or --dataset-root at a directory containing "
                f"{_CFG['audio_subdir']}/ and {_CFG['manifests_subdir']}/."
            )

    def is_built(self, recipe_dir: str | Path, **_) -> bool:
        data_dir = Path(recipe_dir).resolve() / _CFG["data_path"]
        outputs = list(_CFG["manifest_paths"].values()) + [
            _CFG["vocab_path"],
            _CFG["vocab_meta_path"],
        ]
        return all((data_dir / relpath).is_file() for relpath in outputs)

    def build(
        self,
        recipe_dir: str | Path,
        dataset_root: str | Path | None = None,
        seed: int | None = None,
        base_vocab_path: str | Path | None = None,
        **_,
    ) -> None:
        root = resolve_dataset_root(dataset_root)
        seed = int(seed if seed is not None else _CFG["seed"])
        data_dir = Path(recipe_dir).resolve() / _CFG["data_path"]

        base_vocab_path = base_vocab_path or _CFG["base_vocab_path"]
        if base_vocab_path is None:
            raise ValueError(
                "base_vocab_path is required: point it at the pretrained char "
                "vocab to extend (builder.base_vocab_path in config.yaml, or "
                "--base-vocab-path on the CLI). The two new tokens are appended "
                "at the end so every pretrained token id is unchanged."
            )
        base_vocab_path = Path(base_vocab_path)
        if not base_vocab_path.is_file():
            raise FileNotFoundError(f"base vocab not found: {base_vocab_path}")
        base_tokens = [
            line.rstrip("\n")
            for line in base_vocab_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        extended = extend_vocab(base_tokens)
        charset = vocab_charset(extended)

        manifests = root / _CFG["manifests_subdir"]
        recordings = load_recordings(
            manifests / "recordings.jsonl.gz", audio_subdir=_CFG["audio_subdir"]
        )
        supervisions = load_supervisions(
            manifests / "supervisions.jsonl.gz", recordings
        )
        session_ids = sorted(set(recordings) & set(supervisions))
        skipped_no_speech = sorted(set(recordings) - set(supervisions))

        splits = split_sessions(session_ids, _CFG["split_ratios"], seed)
        session_split = {sid: split for split, ids in splits.items() for sid in ids}

        writers = {}
        for split, relpath in _CFG["manifest_paths"].items():
            path = data_dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[split] = path.open("w", encoding="utf-8")

        stats = {split: WindowingStats() for split in writers}
        durations: dict[str, list[float]] = {split: [] for split in writers}
        turns_per_window: dict[str, list[int]] = {split: [] for split in writers}
        speakers: dict[str, set[str]] = {split: set() for split in writers}
        overlap_time = speech_time = 0.0
        dropped_empty_turns = 0

        try:
            for sid in session_ids:
                split = session_split[sid]
                sups = supervisions[sid]
                speakers[split] |= session_speakers(sups)
                turns = merge_turns(sups, _CFG["merge_gap"])
                normalized = []
                for turn in turns:
                    text = normalize_text(turn.text, charset)
                    if not text:
                        dropped_empty_turns += 1
                        continue
                    normalized.append(dataclasses.replace(turn, text=text))
                ov, sp = overlap_and_speech_time(normalized)
                overlap_time += ov
                speech_time += sp
                records, session_stats = build_windows(
                    sid,
                    recordings[sid],
                    normalized,
                    window_min=_CFG["window_min"],
                    window_max=_CFG["window_max"],
                    silence_min=_CFG["silence_min"],
                    tail_min=_CFG["tail_min"],
                    rng=random.Random(f"{seed}:window:{sid}"),
                )
                stats[split].merge(session_stats)
                for record in records:
                    writers[split].write(json.dumps(to_json(record)) + "\n")
                    durations[split].append(record.duration)
                    turns_per_window[split].append(len(record.turns))
        finally:
            for f in writers.values():
                f.close()

        vocab_path = data_dir / _CFG["vocab_path"]
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text("\n".join(extended) + "\n", encoding="utf-8")
        meta = {
            "base_vocab_path": str(base_vocab_path),
            "base_size": len(base_tokens),
            "new_tokens": {
                token: len(base_tokens) + i for i, token in enumerate(NEW_TOKENS)
            },
            "total_size": len(extended),
        }
        (data_dir / _CFG["vocab_meta_path"]).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        print(f"SSSD build summary (seed={seed}, root={root})")
        print(f"  sessions: {len(session_ids)}" + (
            f" (+{len(skipped_no_speech)} skipped, no supervisions)"
            if skipped_no_speech
            else ""
        ))
        print(f"  turns dropped empty after normalization: {dropped_empty_turns}")
        ratio = overlap_time / speech_time if speech_time else 0.0
        print(f"  overlap ratio (>=2 channels active / any active): {ratio:.3f}")
        for split in ("train", "valid", "test"):
            st = stats[split]
            print(
                f"  {split}: {st.n_windows} windows over {len(splits[split])} sessions; "
                f"dropped {st.dropped_span_sec:.1f}s unbreakable spans, "
                f"{st.dropped_tail_sec:.1f}s tails, {st.dropped_empty_windows} empty windows"
            )
            print(f"    duration[s]: {_distribution(durations[split])}")
            print(f"    turns/window: {_distribution(turns_per_window[split])}")
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = len(speakers[a] & speakers[b])
            print(
                f"  speaker overlap {a}({len(speakers[a])}) & {b}({len(speakers[b])}): "
                f"{shared}"
            )
        print(f"  vocab: {meta['total_size']} tokens, new ids {meta['new_tokens']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=SSSDBuilder.__doc__)
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="recipe root; outputs go to <recipe-dir>/data/",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-vocab-path", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if outputs exist"
    )
    args = parser.parse_args()

    builder = SSSDBuilder()
    builder.prepare_source(recipe_dir=args.recipe_dir, dataset_root=args.dataset_root)
    if builder.is_built(recipe_dir=args.recipe_dir) and not args.force:
        print(f"Already built under {args.recipe_dir}/data; use --force to rebuild.")
        return
    builder.build(
        recipe_dir=args.recipe_dir,
        dataset_root=args.dataset_root,
        seed=args.seed,
        base_vocab_path=args.base_vocab_path,
    )


if __name__ == "__main__":
    main()
