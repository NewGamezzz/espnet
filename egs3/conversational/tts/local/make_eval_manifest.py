"""Freeze an inference config's window selection and prompt picks into a
shareable evaluation manifest, and optionally slice it into equal parts.

The manifest is the artifact you hand a collaborator so that "the same test
set with the same audio prompt per sample" is a file rather than a seed plus
a matching checkout.  See ``src/eval_manifest.py`` for the format and why
the seed alone is not enough.

Reads no audio: the window manifest carries every turn span the prompt
ladder needs, so this runs on a login node in seconds.

Slicing exists because ``src/inference.py`` has no shard support (that lives
only in the chunked path).  A slice is the header plus a contiguous run of
window rows, which is itself a valid manifest - so N slices are N ordinary
jobs, and no sampling semantics change.  Merge their output trees before
``measure``.

Usage:
    python -m egs3.conversational.tts.local.make_eval_manifest \
        --inference-config conf/inference_conversational.yaml \
        --out data/eval/sssd_test_2spk_v1.jsonl [--slices 8]

``--inference-config`` supplies the split, the selection filter and the
prompt/sampling blocks; override any of them on the command line with
``--set key=value`` (dotted OmegaConf paths), e.g.
``--set dataset.split=test --set selection.num_windows=null``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[4]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from egs3.conversational.tts.src.eval_manifest import (  # noqa: E402
    build_eval_manifest,
    write_eval_manifest,
)


def slice_rows(rows: list, n_slices: int) -> list[list]:
    """Contiguous, near-equal slices, preserving manifest order.

    Contiguous rather than round-robin so a slice is a readable range of the
    manifest and a failed slice is obvious from the window ids in its log.
    """
    if n_slices < 1:
        raise ValueError(f"--slices must be >= 1, got {n_slices}")
    if n_slices > len(rows):
        raise ValueError(
            f"--slices {n_slices} exceeds the {len(rows)} windows in the manifest"
        )
    base, extra = divmod(len(rows), n_slices)
    out, start = [], 0
    for i in range(n_slices):
        size = base + (1 if i < extra else 0)
        out.append(rows[start : start + size])
        start += size
    assert start == len(rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inference-config", required=True, type=Path)
    ap.add_argument("--training-config", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--slices",
        type=int,
        default=1,
        help="also write <out stem>.<i>of<N>.jsonl slices for sharded runs",
    )
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="dotted OmegaConf override, e.g. dataset.split=test",
    )
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.inference_config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    training_config = (
        OmegaConf.load(args.training_config) if args.training_config else None
    )

    header, rows = build_eval_manifest(cfg, training_config=training_config)
    write_eval_manifest(args.out, header, rows)
    print(json.dumps(header, indent=2))
    print(f"wrote {args.out} ({len(rows)} windows)")
    if header["num_skipped"]:
        # Loud on purpose: the skipped windows are NOT in the test set, so
        # any "N windows" claim downstream must use num_windows, not the
        # eligible count.
        print(
            f"NOTE {header['num_skipped']} of {header['num_eligible']} "
            "eligible windows have no leakage-free prompt on some channel "
            "and are excluded"
        )

    if args.slices > 1:
        for i, part in enumerate(slice_rows(rows, args.slices)):
            path = args.out.with_suffix("")
            path = path.parent / f"{path.name}.{i}of{args.slices}.jsonl"
            part_header = dict(header)
            part_header["num_windows"] = len(part)
            part_header["slice_index"] = i
            part_header["slice_count"] = args.slices
            write_eval_manifest(path, part_header, part)
            print(f"  slice {i}: {len(part):5d} windows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
