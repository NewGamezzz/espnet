"""Frozen evaluation manifests: a shareable recording of WHICH windows an
infer run scores and WHICH turn each channel's prompt is cut from.

Why this exists
---------------
``src/inference.py`` already picks both of those deterministically - windows
by a seeded capped draw, prompt turns by a seeded relaxation ladder - so a
run is reproducible by anyone holding the same config.  It is NOT
reproducible by anyone holding a *different checkout*: the pick is a
function of the code (pool construction, ladder tiers, RNG key format), of
the frozen window manifest, and of the corpus files.  A collaborator who
reruns "the same eval" against any of those at a different revision gets a
silently different test set.

A frozen manifest replaces that chain of assumptions with data.  It names
the windows and pins each channel's prompt as an explicit session-absolute
span, so "the same setting, the same audio prompt for each sample" is a file
you can hand over rather than a seed you have to trust.

Format (JSONL, one object per line)
-----------------------------------
Line 1 is a header, ``record_type: "header"``, carrying provenance: manifest
version, split, the source window manifest and its md5, and the selection /
prompt / sampling blocks that produced the rows.  Every later line is a
window, ``record_type: "window"``::

    {"record_type": "window", "window_id": ..., "session_id": ...,
     "t0": ..., "t1": ...,
     "prompts": [{"channel": 0, "start": ..., "end": ...}, ...]}

``session_id`` / ``t0`` / ``t1`` are ECHOES, not inputs: the window is
resolved by ``window_id`` alone and the echoes are then checked against the
split, so a manifest built against different data fails loudly instead of
silently scoring a different window.  Prompt spans are likewise resolved by
exact match against the session's turn pool, so a corpus that has moved
under the manifest is an error rather than a near-miss.

Line-oriented on purpose: a slice of the rows plus the header is itself a
valid manifest, which is how a long run is split across jobs without adding
any sharding semantics to the sampling path.

This module is the WRITER and the LOADER.  The consumer is
``src/inference.py``, which imports the loader; the builder imports
``inference`` lazily so the two can depend on each other without a cycle.
Nothing here is imported by the training path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MANIFEST_VERSION = 1
HEADER_TYPE = "header"
WINDOW_TYPE = "window"

# Spans round-trip through JSON at 6 decimals (``_prompt_turn_meta``'s
# convention), so matching is tolerant to exactly that quantum and no more.
SPAN_TOL = 1e-6


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spans_match(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= SPAN_TOL


def build_eval_manifest(inference_config, *, training_config=None):
    """Freeze an inference config's seeded selection into ``(header, rows)``.

    Runs the REAL selection and the REAL ladder - imported from
    ``src/inference.py``, not reimplemented - so the manifest is a recording
    of current behaviour rather than a second implementation of it.  Reads
    no audio: everything needed lives in the window manifest.
    """
    from omegaconf import OmegaConf

    from egs3.conversational.tts.src.generation import build_dataset
    from egs3.conversational.tts.src.inference import (
        _build_turn_pools,
        _select_indices,
        _select_prompt_turn,
        load_excluded_spans,
    )

    cfg = inference_config
    if training_config is None:
        train_path = Path(cfg.training_config)
        if not train_path.is_absolute():
            train_path = Path(cfg.get("recipe_dir", ".")) / train_path
        training_config = OmegaConf.load(train_path)

    dataset = build_dataset(
        training_config,
        cfg.dataset.split,
        inference=True,
        manifest_path=cfg.dataset.get("manifest_path"),
        dataset_root=cfg.dataset.get("dataset_root"),
    )
    pools = _build_turn_pools(dataset.records)
    indices = _select_indices(dataset.records, cfg.selection)

    prompt_cfg = cfg.prompt
    turn_min = float(prompt_cfg.get("turn_min_sec", 2.0))
    turn_max = float(prompt_cfg.get("turn_max_sec", 10.0))
    prompt_seed = prompt_cfg.get("seed", 0)
    solo_guard = float(prompt_cfg.get("solo_guard_sec", 0.0) or 0.0)
    excluded_by_session = (
        load_excluded_spans(prompt_cfg.exclude_spans)
        if prompt_cfg.get("exclude_spans")
        else {}
    )

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for idx in indices:
        record = dataset.records[idx]
        pool_turns = pools.get(record.session_id, [])
        selected = []
        excluded = excluded_by_session.get(record.session_id, frozenset())
        for ch in record.row_channels:  # SOURCE channels, row order
            turn = _select_prompt_turn(
                pool_turns,
                ch,
                record.t0,
                record.t1,
                turn_min,
                turn_max,
                prompt_seed,
                record.window_id,
                solo_guard=solo_guard,
                excluded=excluded,
            )
            if turn is None:
                selected = None
                break
            selected.append(turn)
        if selected is None:
            # Same skip rule as the infer stage: a window no channel can be
            # prompted for leakage-free is not in the test set at all.
            n_skipped += 1
            continue
        rows.append(
            {
                "record_type": WINDOW_TYPE,
                "window_id": record.window_id,
                "session_id": record.session_id,
                "t0": round(float(record.t0), 6),
                "t1": round(float(record.t1), 6),
                "source_channels": list(record.row_channels),
                "prompts": [
                    {
                        "channel": int(t.channel),
                        "start": round(float(t.start), 6),
                        "end": round(float(t.end), 6),
                    }
                    for t in selected
                ],
            }
        )

    source = getattr(dataset, "manifest_path", None)
    header = {
        "record_type": HEADER_TYPE,
        "manifest_version": MANIFEST_VERSION,
        "split": str(cfg.dataset.split),
        "source_manifest": str(source) if source else None,
        "source_manifest_md5": _md5(Path(source)) if source else None,
        "num_windows": len(rows),
        "num_skipped": n_skipped,
        "num_eligible": len(indices),
        "selection": OmegaConf.to_container(cfg.selection, resolve=True),
        "prompt": OmegaConf.to_container(cfg.prompt, resolve=True),
        "sampling": OmegaConf.to_container(cfg.sampling, resolve=True),
    }
    # The manifest pins the selection, so carrying the draw knobs that are
    # now inert would invite someone to "fix" them and expect an effect.
    header["selection"].pop("manifest", None)
    return header, rows


def write_eval_manifest(path, header: dict, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in [header, *rows]:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_eval_manifest(path):
    """Return ``(header, rows)``; raise ``ValueError`` on anything malformed.

    Structural validation only - whether the rows agree with a particular
    split is decided later, against that split.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"eval manifest not found: {path}")
    objs = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not JSON ({exc})") from exc
    if not objs:
        raise ValueError(f"{path}: empty eval manifest")
    header = objs[0]
    if header.get("record_type") != HEADER_TYPE:
        raise ValueError(
            f"{path}: first line must be the {HEADER_TYPE!r} record, got "
            f"{header.get('record_type')!r}"
        )
    version = header.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest_version {version!r}, this code reads "
            f"{MANIFEST_VERSION}"
        )
    rows = objs[1:]
    seen: set[str] = set()
    for i, row in enumerate(rows, 1):
        if row.get("record_type") != WINDOW_TYPE:
            raise ValueError(
                f"{path}: row {i} has record_type "
                f"{row.get('record_type')!r}, expected {WINDOW_TYPE!r}"
            )
        wid = row.get("window_id")
        if not wid:
            raise ValueError(f"{path}: row {i} has no window_id")
        if wid in seen:
            raise ValueError(f"{path}: duplicate window_id {wid!r}")
        seen.add(wid)
        if not row.get("prompts"):
            raise ValueError(f"{path}: {wid} has no prompts")
    return header, rows
