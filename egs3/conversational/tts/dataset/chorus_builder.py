"""NSF Chorus dataset builder: multi-party (4-8 channel) session manifests.

Runs AFTER the vocab exists in ``data/tokens/vocab.txt`` (copied from the
stage-1 checkout for stage 2, or written by the SSSD build): transcripts are
normalized against that charset.  ``prepare_source`` runs the one-time
per-speaker wav -> N-channel FLAC merge (a short compute job: 16 GB in);
``build`` cleans the markup, merges turns against measured FLAC durations,
and writes one ``SessionRecord`` per meeting into the corpus's OWN splits
(train/dev/eval -> train/valid/test).

Chorus sessions come out as ordinary N-channel conversations; the planner,
collator, and TAC exchanges are channel-count generic, so nothing downstream
of the manifest changes.  Memory is the one N-dependent concern: the stage-2
training config caps Chorus windows (see conf/training_stage2_chorus_h100.yaml).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from .preprocessing.chorus import (
    clean_chorus_supervisions,
    load_chorus_manifest,
    measured_durations_nch,
    merge_all,
    merged_relpath,
)
from .preprocessing.sessions import SessionRecord, write_session_manifest
from .preprocessing.sssd import merge_turns
from .preprocessing.text import normalize_text, vocab_charset


def _load_configs() -> tuple[dict, dict]:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        cfg = load_config_with_defaults(str(config_path), resolve=False)
    return cfg["chorus_builder"], cfg["builder"]


_CFG, _SSSD_CFG = _load_configs()


def resolve_chorus_root(explicit: str | Path | None = None) -> Path:
    """Corpus root resolution: explicit > $CHORUS_ROOT > config."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("CHORUS_ROOT"):
        return Path(os.environ["CHORUS_ROOT"])
    return Path(_CFG["dataset_root"])


def resolve_flac_dir(explicit: str | Path | None = None) -> Path:
    """Merged-FLAC dir resolution: explicit > $CHORUS_FLAC_DIR > config."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("CHORUS_FLAC_DIR"):
        return Path(os.environ["CHORUS_FLAC_DIR"])
    return Path(_CFG["flac_dir"])


class ChorusBuilder(DatasetBuilder):
    """Prepare NSF Chorus multi-party session manifests for mixed training."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        chorus_flac_dir: str | Path | None = None,
        **_,
    ) -> bool:
        root = resolve_chorus_root(dataset_root)
        manifest = root / _CFG["manifest_file"]
        if not manifest.is_file():
            return False
        fdir = resolve_flac_dir(chorus_flac_dir)
        meetings = load_chorus_manifest(manifest)
        return all((fdir / merged_relpath(m)).is_file() for m in meetings.values())

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        chorus_flac_dir: str | Path | None = None,
        **_,
    ) -> None:
        root = resolve_chorus_root(dataset_root)
        manifest = root / _CFG["manifest_file"]
        if not manifest.is_file():
            raise RuntimeError(
                f"NSF Chorus corpus not found at {root}. Point dataset_root "
                "(config), $CHORUS_ROOT, or --dataset-root at the directory "
                f"containing {_CFG['manifest_file']} and <split>/<MTG_id>/*.wav."
            )
        fdir = resolve_flac_dir(chorus_flac_dir)
        meetings = load_chorus_manifest(manifest)
        written = merge_all(
            meetings,
            root,
            fdir,
            ffmpeg=str(_CFG["ffmpeg"]),
            workers=int(_CFG["merge_workers"]),
        )
        print(f"Chorus merge: {written} new N-channel FLACs -> {fdir}")

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
        chorus_flac_dir: str | Path | None = None,
        seed: int | None = None,
        **_,
    ) -> None:
        root = resolve_chorus_root(dataset_root)
        fdir = resolve_flac_dir(chorus_flac_dir)
        data_dir = Path(recipe_dir).resolve() / _SSSD_CFG["data_path"]

        vocab_path = data_dir / _SSSD_CFG["vocab_path"]
        if not vocab_path.is_file():
            raise RuntimeError(
                f"extended vocab not found at {vocab_path}; copy the stage-1 "
                "checkout's data/tokens/ or run the SSSD build first. Chorus "
                "normalization must use the same charset."
            )
        charset = vocab_charset(vocab_path.read_text(encoding="utf-8").splitlines())

        meetings = load_chorus_manifest(root / _CFG["manifest_file"])
        durations = measured_durations_nch(meetings, fdir)
        split_map = dict(_CFG["split_map"])

        session_records: dict[str, list[SessionRecord]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        turn_seconds = {split: 0.0 for split in session_records}
        speakers: dict[str, set[str]] = {split: set() for split in session_records}
        channel_counts: dict[str, list[int]] = {split: [] for split in session_records}
        dropped_empty_turns = 0
        dropped_out_of_range_turns = 0
        dropped_benign_utts = 0
        dropped_unintelligible_utts = 0
        exclusion_span_seconds = 0.0
        for mid in sorted(meetings):
            m = meetings[mid]
            if m.split not in split_map:
                raise ValueError(
                    f"meeting {mid!r} has split {m.split!r}, not in split_map "
                    f"{sorted(split_map)}"
                )
            split = split_map[m.split]
            speakers[split] |= set(m.speakers)
            channel_counts[split].append(m.num_channels)
            kept, spans, n_benign = clean_chorus_supervisions(m.utterances)
            dropped_benign_utts += n_benign
            dropped_unintelligible_utts += len(spans)
            exclusion_span_seconds += sum(b - a for a, b in spans)
            turns = merge_turns(kept, float(_CFG["merge_gap"]))
            normalized = []
            duration = durations[mid]
            for turn in turns:
                # Turns must never overrun the measured audio.
                end = min(turn.end, duration)
                if end <= turn.start:
                    dropped_out_of_range_turns += 1
                    continue
                text = normalize_text(turn.text, charset)
                if not text:
                    dropped_empty_turns += 1
                    continue
                normalized.append(dataclasses.replace(turn, text=text, end=end))
            turn_seconds[split] += sum(t.end - t.start for t in normalized)
            session_records[split].append(
                SessionRecord(
                    session_id=mid,
                    audio_relpath=merged_relpath(m),
                    num_channels=m.num_channels,
                    sample_rate=m.sample_rate,
                    duration=duration,
                    turns=tuple(normalized),
                    exclusion_spans=tuple(spans),
                )
            )

        for split, relpath in _CFG["manifest_paths"].items():
            write_session_manifest(data_dir / relpath, session_records[split])

        print(f"Chorus build summary (root={root})")
        print(f"  meetings: {len(meetings)}")
        print(f"  turns dropped empty after normalization: {dropped_empty_turns}")
        print(
            f"  turns dropped out-of-range (end <= start): {dropped_out_of_range_turns}"
        )
        print(f"  utterances dropped benign (tag-only): {dropped_benign_utts}")
        print(
            "  utterances dropped unintelligible (exclusion spans, "
            f"{exclusion_span_seconds:.1f}s total, dropped online by the "
            f"planner): {dropped_unintelligible_utts}"
        )
        for split in ("train", "valid", "test"):
            counts = sorted(set(channel_counts[split]))
            print(
                f"  {split}: {len(session_records[split])} meetings, "
                f"{turn_seconds[split] / 3600:.2f}h turns, channels {counts}"
            )
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = len(speakers[a] & speakers[b])
            print(
                f"  speaker overlap {a}({len(speakers[a])}) & {b}({len(speakers[b])}): "
                f"{shared}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NSF Chorus session manifests")
    parser.add_argument(
        "--recipe-dir", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--chorus-flac-dir", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only run the wav -> N-channel FLAC merge (the compute step)",
    )
    args = parser.parse_args()
    b = ChorusBuilder()
    kw = dict(
        recipe_dir=args.recipe_dir,
        dataset_root=args.dataset_root,
        chorus_flac_dir=args.chorus_flac_dir,
    )
    if not b.is_source_prepared(**kw):
        b.prepare_source(**kw)
    if not args.prepare_only:
        b.build(**kw)


if __name__ == "__main__":
    main()
