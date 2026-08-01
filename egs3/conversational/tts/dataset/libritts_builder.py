"""LibriTTS dataset builder: single-speaker window manifests for the mix.

Runs AFTER the SSSD build (``python -m egs3.conversational.tts.dataset.builder``):
it normalizes transcripts against the charset of the extended vocab that build
wrote, so token coverage can never diverge between corpora.  Emits
``manifest/libritts_{train,valid}.jsonl`` in the exact ``WindowRecord`` schema
``ConversationDataset`` already consumes; no new dataset class exists.

The corpus directory is treated as strictly read-only.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from .builder import _distribution
from .preprocessing.libritts import scan_subset, subsample_to_hours, utterance_record
from .preprocessing.text import normalize_text, vocab_charset
from .preprocessing.windows import write_window_manifest


def _load_configs() -> tuple[dict, dict]:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        cfg = load_config_with_defaults(str(config_path), resolve=False)
    return cfg["libritts_builder"], cfg["builder"]


_CFG, _SSSD_CFG = _load_configs()


def resolve_libritts_root(explicit: str | Path | None = None) -> Path:
    """Corpus root resolution: explicit argument > ``$LIBRITTS_ROOT`` >
    ``libritts_builder.dataset_root`` in config.yaml."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("LIBRITTS_ROOT"):
        return Path(os.environ["LIBRITTS_ROOT"])
    return Path(_CFG["dataset_root"])


class LibriTTSBuilder(DatasetBuilder):
    """Prepare LibriTTS single-speaker window manifests for mixed training."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        **_,
    ) -> bool:
        root = resolve_libritts_root(dataset_root)
        subsets = list(_CFG["train_subsets"]) + [_CFG["valid_subset"]]
        return all((root / subset).is_dir() for subset in subsets)

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        **_,
    ) -> None:
        if not self.is_source_prepared(dataset_root=dataset_root):
            root = resolve_libritts_root(dataset_root)
            raise RuntimeError(
                f"LibriTTS corpus not found at {root}. The corpus is externally "
                "provisioned and read-only; point dataset_root (config), "
                "$LIBRITTS_ROOT, or --dataset-root at a directory containing "
                f"{', '.join(_CFG['train_subsets'])} and {_CFG['valid_subset']}."
            )

    def is_built(self, recipe_dir: str | Path, **_) -> bool:
        data_dir = Path(recipe_dir).resolve() / _SSSD_CFG["data_path"]
        return all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def build(
        self,
        recipe_dir: str | Path,
        dataset_root: str | Path | None = None,
        seed: int | None = None,
        **_,
    ) -> None:
        root = resolve_libritts_root(dataset_root)
        seed = int(seed if seed is not None else _CFG["seed"])
        data_dir = Path(recipe_dir).resolve() / _SSSD_CFG["data_path"]

        vocab_path = data_dir / _SSSD_CFG["vocab_path"]
        if not vocab_path.is_file():
            raise RuntimeError(
                f"extended vocab not found at {vocab_path}; run the SSSD build "
                "first (python -m egs3.conversational.tts.dataset.builder). "
                "LibriTTS normalization must use the same charset."
            )
        charset = vocab_charset(vocab_path.read_text(encoding="utf-8").splitlines())

        expected_rate = int(_CFG["sample_rate"])
        min_duration = float(_CFG["min_duration"])
        print(f"LibriTTS build summary (seed={seed}, root={root})")
        splits = (
            ("train", list(_CFG["train_subsets"]), None),
            ("valid", [_CFG["valid_subset"]], float(_CFG["valid_hours"])),
        )
        scan_workers = int(_CFG.get("scan_workers", 0)) or min(32, os.cpu_count() or 8)
        for split, subsets, hours_cap in splits:
            entries = []
            for subset in subsets:
                entries.extend(scan_subset(root, subset, workers=scan_workers))
            # Header probes are tiny independent reads dominated by filesystem
            # latency (hours if serial on a cold parallel filesystem); the
            # thread pool keeps input order, so output stays deterministic.
            with ThreadPoolExecutor(max_workers=scan_workers) as pool:
                infos = list(
                    pool.map(
                        lambda entry: sf.info(str(root / entry.audio_relpath)),
                        entries,
                    )
                )
            pairs = []  # (entry, duration)
            dropped_short = 0
            for entry, info in zip(entries, infos):
                if info.samplerate != expected_rate:
                    raise RuntimeError(
                        f"{entry.audio_relpath}: sample rate "
                        f"{info.samplerate} != expected {expected_rate}"
                    )
                duration = info.frames / info.samplerate
                if duration < min_duration:
                    dropped_short += 1
                    continue
                pairs.append((entry, duration))
            if hours_cap is not None:
                pairs = subsample_to_hours(pairs, hours_cap, seed)

            records = []
            dropped_empty = 0
            for entry, duration in pairs:
                text = normalize_text(entry.text, charset)
                if not text:
                    dropped_empty += 1
                    continue
                records.append(utterance_record(entry, duration, expected_rate, text))
            n = write_window_manifest(data_dir / _CFG["manifest_paths"][split], records)
            total_h = sum(r.duration for r in records) / 3600
            print(
                f"  {split}: {n} utterances ({total_h:.1f}h) from "
                f"{', '.join(subsets)}; dropped {dropped_short} short "
                f"(< {min_duration:g}s), {dropped_empty} empty after "
                "normalization"
            )
            print(f"    duration[s]: {_distribution([r.duration for r in records])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=LibriTTSBuilder.__doc__)
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="recipe root; manifests go to <recipe-dir>/data/manifest/",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if outputs exist"
    )
    args = parser.parse_args()

    builder = LibriTTSBuilder()
    builder.prepare_source(recipe_dir=args.recipe_dir, dataset_root=args.dataset_root)
    if builder.is_built(recipe_dir=args.recipe_dir) and not args.force:
        print(f"Already built under {args.recipe_dir}/data; use --force to rebuild.")
        return
    builder.build(
        recipe_dir=args.recipe_dir,
        dataset_root=args.dataset_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
