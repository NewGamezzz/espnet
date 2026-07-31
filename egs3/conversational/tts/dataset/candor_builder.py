"""CANDOR dataset builder: conversational window manifests for the mix.

Runs AFTER the SSSD build (``python -m egs3.conversational.tts.dataset.builder``):
transcripts are normalized against the extended vocab that build wrote.
``prepare_source`` verifies the read-only corpus and runs the one-time
mp3 -> FLAC transcode (a COMPUTE JOB on the cluster: ~60-70 core-hours,
~240 GB); ``build`` is cheap - it windows the transcripts against measured
FLAC durations and writes ``manifest/candor_{train,valid,test}.jsonl`` in the
exact ``WindowRecord`` schema ``ConversationDataset`` already consumes.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from .builder import _distribution, split_sessions
from .preprocessing.candor import (
    load_candor_recordings,
    measured_durations,
    transcode_all,
)
from .preprocessing.sssd import load_supervisions, merge_turns, session_speakers
from .preprocessing.text import normalize_text, vocab_charset
from .preprocessing.windows import (
    WindowingStats,
    build_windows,
    write_window_manifest,
)


def _load_configs() -> tuple[dict, dict]:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        cfg = load_config_with_defaults(str(config_path), resolve=False)
    return cfg["candor_builder"], cfg["builder"]


_CFG, _SSSD_CFG = _load_configs()


def resolve_candor_root(explicit: str | Path | None = None) -> Path:
    """Corpus root resolution: explicit > $CANDOR_ROOT > config."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("CANDOR_ROOT"):
        return Path(os.environ["CANDOR_ROOT"])
    return Path(_CFG["dataset_root"])


def resolve_flac_dir(explicit: str | Path | None = None) -> Path:
    """Transcode dir resolution: explicit > $CANDOR_FLAC_DIR > config."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("CANDOR_FLAC_DIR"):
        return Path(os.environ["CANDOR_FLAC_DIR"])
    return Path(_CFG["flac_dir"])


def _manifest_paths(root: Path) -> tuple[Path, Path]:
    manifests = root / _CFG["manifests_subdir"]
    return (
        manifests / "candor_recordings.jsonl.gz",
        manifests / f"candor_supervisions_{_CFG['variant']}.jsonl.gz",
    )


class CandorBuilder(DatasetBuilder):
    """Prepare CANDOR conversational window manifests for mixed training."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        flac_dir: str | Path | None = None,
        **_,
    ) -> bool:
        root = resolve_candor_root(dataset_root)
        rec_path, sup_path = _manifest_paths(root)
        if not (rec_path.is_file() and sup_path.is_file()):
            return False
        fdir = resolve_flac_dir(flac_dir)
        recordings = load_candor_recordings(rec_path)
        return all((fdir / rec.audio_relpath).is_file() for rec in recordings.values())

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        flac_dir: str | Path | None = None,
        **_,
    ) -> None:
        root = resolve_candor_root(dataset_root)
        rec_path, sup_path = _manifest_paths(root)
        if not (rec_path.is_file() and sup_path.is_file()):
            raise RuntimeError(
                f"CANDOR corpus not found at {root}. The corpus is externally "
                "provisioned and read-only; point dataset_root (config), "
                "$CANDOR_ROOT, or --dataset-root at a directory containing "
                f"{_CFG['manifests_subdir']}/ and candor_data/."
            )
        fdir = resolve_flac_dir(flac_dir)
        recordings = load_candor_recordings(rec_path)
        written = transcode_all(
            recordings,
            root,
            fdir,
            ffmpeg=str(_CFG["ffmpeg"]),
            workers=int(_CFG["transcode_workers"]),
        )
        print(f"CANDOR transcode: {written} new FLACs -> {fdir}")

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
        flac_dir: str | Path | None = None,
        seed: int | None = None,
        **_,
    ) -> None:
        root = resolve_candor_root(dataset_root)
        fdir = resolve_flac_dir(flac_dir)
        seed = int(seed if seed is not None else _CFG["seed"])
        data_dir = Path(recipe_dir).resolve() / _SSSD_CFG["data_path"]

        vocab_path = data_dir / _SSSD_CFG["vocab_path"]
        if not vocab_path.is_file():
            raise RuntimeError(
                f"extended vocab not found at {vocab_path}; run the SSSD build "
                "first (python -m egs3.conversational.tts.dataset.builder). "
                "CANDOR normalization must use the same charset."
            )
        charset = vocab_charset(vocab_path.read_text(encoding="utf-8").splitlines())

        rec_path, sup_path = _manifest_paths(root)
        recordings = load_candor_recordings(rec_path)
        durations = measured_durations(recordings, fdir)
        recordings = {
            cid: dataclasses.replace(rec, duration=durations[cid])
            for cid, rec in recordings.items()
        }
        supervisions = load_supervisions(sup_path, recordings)
        session_ids = sorted(set(recordings) & set(supervisions))

        splits = split_sessions(session_ids, _CFG["split_ratios"], seed)
        session_split = {sid: split for split, ids in splits.items() for sid in ids}

        records_by_split = {split: [] for split in _CFG["manifest_paths"]}
        stats = {split: WindowingStats() for split in records_by_split}
        spk_windows = {split: Counter() for split in records_by_split}
        spk_seconds = {split: defaultdict(float) for split in records_by_split}
        speakers: dict[str, set[str]] = {split: set() for split in records_by_split}
        dropped_empty_turns = 0
        dropped_out_of_range_turns = 0
        for sid in session_ids:
            split = session_split[sid]
            speakers[split] |= session_speakers(supervisions[sid])
            turns = merge_turns(supervisions[sid], _CFG["merge_gap"])
            normalized = []
            for turn in turns:
                # A supervision starting past the measured audio end clamps
                # to a negative span in load_supervisions (duration = min(...,
                # rec.duration - start) < 0); drop rather than emit a window
                # turn whose end is at or before its own start.
                if turn.end <= turn.start:
                    dropped_out_of_range_turns += 1
                    continue
                text = normalize_text(turn.text, charset)
                if not text:
                    dropped_empty_turns += 1
                    continue
                normalized.append(dataclasses.replace(turn, text=text))
            records, session_stats = build_windows(
                sid,
                recordings[sid],
                normalized,
                window_min=_CFG["window_min"],
                window_max=_CFG["window_max"],
                boundary_guard=_CFG["boundary_guard"],
                tail_min=_CFG["tail_min"],
                rng=random.Random(f"{seed}:window:{sid}"),
                trim_to_turns=_CFG["trim_to_turns"],
                min_coverage=_CFG["min_coverage"],
                snap_start_to_turn=_CFG["snap_start_to_turn"],
            )
            stats[split].merge(session_stats)
            records_by_split[split].extend(records)
            for record in records:
                spk_windows[split][record.num_active_speakers] += 1
                spk_seconds[split][record.num_active_speakers] += record.duration

        print(
            f"CANDOR build summary (seed={seed}, variant={_CFG['variant']}, "
            f"root={root})"
        )
        print(f"  sessions: {len(session_ids)}")
        print(f"  turns dropped empty after normalization: {dropped_empty_turns}")
        print(
            "  turns dropped out-of-range (clamped end <= start): "
            f"{dropped_out_of_range_turns}"
        )
        for split, records in records_by_split.items():
            n = write_window_manifest(data_dir / _CFG["manifest_paths"][split], records)
            st = stats[split]
            hours = sum(r.duration for r in records) / 3600
            print(
                f"  {split}: {n} windows ({hours:.1f}h) over "
                f"{len(splits[split])} sessions; "
                f"dropped {st.dropped_span_sec / 3600:.1f}h oversized spans, "
                f"{st.dropped_tail_sec:.1f}s tails, "
                f"{st.dropped_empty_windows} empty windows"
            )
            print(
                f"    duration[s]: " f"{_distribution([r.duration for r in records])}"
            )
            by_spk = ", ".join(
                f"{k}spk {spk_windows[split][k]} "
                f"({spk_seconds[split][k] / 3600:.1f}h)"
                for k in sorted(spk_windows[split])
            )
            print(f"    windows by active speakers: {by_spk or 'n=0'}")
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = len(speakers[a] & speakers[b])
            print(
                f"  speaker overlap {a}({len(speakers[a])}) & {b}({len(speakers[b])}): "
                f"{shared}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=CandorBuilder.__doc__)
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="recipe root; manifests go to <recipe-dir>/data/manifest/",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--flac-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if outputs exist"
    )
    args = parser.parse_args()

    builder = CandorBuilder()
    if not builder.is_source_prepared(
        dataset_root=args.dataset_root, flac_dir=args.flac_dir
    ):
        builder.prepare_source(dataset_root=args.dataset_root, flac_dir=args.flac_dir)
    if builder.is_built(recipe_dir=args.recipe_dir) and not args.force:
        print(f"Already built under {args.recipe_dir}/data; use --force to rebuild.")
        return
    builder.build(
        recipe_dir=args.recipe_dir,
        dataset_root=args.dataset_root,
        flac_dir=args.flac_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
