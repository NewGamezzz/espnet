"""Eval battery orchestrator (Task 8): wires manifest + generated-wav
artifacts through diarization, ASR, WER/cpWER, speaker similarity, and
UTMOS into one row per eval window, pools rows into a run-level aggregate,
and drives the ``python -m eval.run_eval`` CLI.

Engine-agnostic by design: this module reads only ``eval.manifest``
entries and wav files on disk (plus the Task 6/7 ``<id>.json`` resume/error
marker) - it never knows or cares whether the audio came from the vLLM
server or the espnet decode path. Every heavy, model-backed callable
(diarization, ASR, speaker embedding, UTMOS) is injected via ``EvalDeps``,
so ``evaluate_record``/``run_battery``/``aggregate`` are pure orchestration
and unit-testable with scripted fakes (see
``eval/tests/test_run_eval.py``).

Lazy-import discipline (binding constraint): importing this module must
never pull in ``torch``/``transformers``/``pyannote`` - every module-level
import below (``eval.diarize``, ``eval.asr``, ``eval.metrics.*``) is
already lazy-internally per its own docstring/hygiene test, so referencing
their functions here is safe. The only place those heavy deps actually
load is ``EvalDeps.__post_init__``'s *real* defaults (``default_embed_fn``,
etc.) - which run when an ``EvalDeps`` is *constructed* without fakes, not
merely imported - and ``main`` when it builds one for the real CLI run.

Aggregation rule (binding project rule, matching
``eval.metrics.wer``'s own docstring): WER is pooled I/D/S counts summed
across rows THEN divided once - never averaged per-row. ``aggregate``'s
pooling of ``wer_concat_counts``/``cpwer_counts`` is the only path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import soundfile as sf

from eval.asr import Word, assign_words, transcribe
from eval.diarize import DiarSegment, diarize, purity
from eval.manifest import load_manifest
from eval.metrics.simo import (
    cluster_cross_similarity,
    default_embed_fn,
    reference_embedding,
    segment_similarities,
)
from eval.metrics.utmos import utmos
from eval.metrics.wer import ErrorCounts, cpwer, wer_concat

_ROW_FIELDS = (
    "example_id",
    "n_clusters",
    "wer_concat_counts",
    "cpwer_counts",
    "cpwer_mapping",
    "mapping_disagrees",
    "sim_own_mean",
    "sim_margin_mean",
    "sim_cross_gt",
    "utmos",
    "purity_gt",
    "duration_s",
    "error",
)


@dataclass
class EvalDeps:
    """Injectable bundle of every heavy, model-backed callable the battery
    needs. Any of ``diarize_fn``/``transcribe_fn``/``embed_fn``/
    ``utmos_fn`` left ``None`` is resolved to the real wrapper in
    ``__post_init__``, bound to ``hf_token``/``device`` - so a plain
    ``EvalDeps()`` (or one built by ``_build_deps`` for the CLI) loads
    every real model, while tests construct one with every callable faked
    and never touch a model at all.
    """

    hf_token: str | None = None
    device: str = "cuda"
    diarize_fn: Callable[[str], list[DiarSegment]] | None = None
    transcribe_fn: Callable[[str], tuple[str, list[Word]]] | None = None
    embed_fn: Callable | None = None
    utmos_fn: Callable[[str], float] | None = None

    def __post_init__(self) -> None:
        if self.diarize_fn is None:
            hf_token, device = self.hf_token, self.device
            self.diarize_fn = lambda wav_path: diarize(
                wav_path, hf_token=hf_token, device=device
            )
        if self.transcribe_fn is None:
            device = self.device
            self.transcribe_fn = lambda wav_path: transcribe(wav_path, device=device)
        if self.embed_fn is None:
            self.embed_fn = default_embed_fn(self.device)
        if self.utmos_fn is None:
            device = self.device
            self.utmos_fn = lambda wav_path: utmos(wav_path, device=device)


def _wav_duration(wav_path: str) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _counts_to_dict(counts: ErrorCounts) -> dict:
    return {
        "hits": counts.hits,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
    }


def _refs_by_speaker(turns: list[dict]) -> dict[str, str]:
    """Concatenate each speaker's turn texts, in turn order, into cpWER's
    ``refs_by_speaker`` shape."""
    refs: dict[str, list[str]] = {}
    for turn in turns:
        refs.setdefault(turn["speaker"], []).append(turn["text"])
    return {speaker: " ".join(texts) for speaker, texts in refs.items()}


def _turns_for_speaker(turns: list[dict], speaker: str) -> list[dict]:
    return [turn for turn in turns if turn["speaker"] == speaker]


def _blank_row(example_id: str, error: str | None = None) -> dict:
    return {
        "example_id": example_id,
        "n_clusters": None,
        "wer_concat_counts": None,
        "cpwer_counts": None,
        "cpwer_mapping": None,
        "mapping_disagrees": None,
        "sim_own_mean": None,
        "sim_margin_mean": None,
        "sim_cross_gt": None,
        "utmos": None,
        "purity_gt": None,
        "duration_s": None,
        "error": error,
    }


def evaluate_record(
    entry: dict, wav_path: str, deps: EvalDeps, *, mode: str = "generated"
) -> dict:
    """Score one manifest entry's audio at ``wav_path`` into a row dict
    (see ``_ROW_FIELDS``).

    ``mode`` (keyword-only; the brief's 3-positional-arg signature stays
    intact for direct calls) controls two mode-dependent fields: Set A's
    ``purity_gt`` (diarization purity of ``wav_path`` against
    ``entry["turns"]``) is only filled when ``mode == "anchor"`` -
    equivalently, only when ``wav_path`` is that entry's own ``gt_wav``, so
    purity is meaningful. Set B's ``sim_cross_gt`` (generated-vs-GT cluster
    similarity) is only filled when ``mode == "generated"`` - scoring
    ``gt_wav`` against itself in anchor mode would be a trivial, meaningless
    self-comparison.

    ``mapping_disagrees`` (Set A only) compares the cpWER-optimal
    cluster->speaker mapping against the similarity-optimal one, but only
    over clusters BOTH sides actually cover - a cluster sim never embedded
    (no segment long enough) is not scored as a disagreement just because
    cpWER still named it. It is ``None`` (not ``False``) when the two sides
    share no cluster to compare.

    Any stage failing (missing/corrupt wav, zero diarization segments -
    ``eval.asr.assign_words`` raises via ``min()`` on an empty segment
    list, guarded here explicitly - etc.) is caught and recorded in
    ``row["error"]``; this function never raises, so ``run_battery`` can
    keep going across a whole manifest.
    """
    row = _blank_row(entry["example_id"])
    try:
        row["duration_s"] = _wav_duration(wav_path)

        if entry["set"] == "librispeech":
            # Single known speaker, no cloning reference: the design
            # mandates WER/UTMOS only, so the (expensive) diarization
            # stage is skipped outright and n_clusters stays None.
            segments = []
        else:
            segments = deps.diarize_fn(wav_path)
            row["n_clusters"] = len({segment.cluster for segment in segments})

        full_text, words = deps.transcribe_fn(wav_path)
        turn_texts = [turn["text"] for turn in entry["turns"]]
        row["wer_concat_counts"] = _counts_to_dict(wer_concat(turn_texts, full_text))

        row["utmos"] = deps.utmos_fn(wav_path)

        if entry["set"] == "sssd":
            if not segments:
                raise ValueError(
                    "zero diarization segments; cannot assign words or score cpWER"
                )
            hyps_by_cluster = assign_words(words, segments)
            refs_by_speaker = _refs_by_speaker(entry["turns"])
            cp = cpwer(refs_by_speaker, hyps_by_cluster)
            row["cpwer_counts"] = _counts_to_dict(cp.counts)
            row["cpwer_mapping"] = cp.mapping

            ref_embs = {
                speaker: reference_embedding(
                    entry["ref_wavs"][speaker],
                    _turns_for_speaker(entry["turns"], speaker),
                    deps.embed_fn,
                )
                for speaker in entry["speakers"]
            }
            sim = segment_similarities(wav_path, segments, ref_embs, deps.embed_fn)
            row["sim_own_mean"] = sim.sim_own_mean
            row["sim_margin_mean"] = sim.margin_mean

            # cpwer.mapping names every cluster, mapping unmatched ones to
            # None; sim.assignment simply omits clusters it never matched -
            # both clusters segment_similarities never embedded (no segment
            # >= its min_sec floor) and clusters its brute-force assignment
            # left unmatched. Drop the None entries so "no mapping" reads
            # the same way on both sides, then compare only over clusters
            # BOTH sides actually cover: a cluster cpWER maps but sim could
            # never embed is not evidence of disagreement, just of sim
            # having less to go on. mapping_disagrees is None (not False)
            # when the two sides share no cluster to compare at all.
            cpwer_mapped = {
                cluster: speaker
                for cluster, speaker in cp.mapping.items()
                if speaker is not None
            }
            common_clusters = set(cpwer_mapped) & set(sim.assignment)
            row["mapping_disagrees"] = (
                any(
                    cpwer_mapped[cluster] != sim.assignment[cluster]
                    for cluster in common_clusters
                )
                if common_clusters
                else None
            )

            if mode == "anchor":
                row["purity_gt"] = purity(segments, entry["turns"])
        elif entry["set"] == "sft":  # Set B: no speaker labels, so no cpWER/sim_own.
            if mode == "generated":
                gt_segments = deps.diarize_fn(entry["gt_wav"])
                row["sim_cross_gt"] = cluster_cross_similarity(
                    wav_path, segments, entry["gt_wav"], gt_segments, deps.embed_fn
                )
    except Exception as exc:  # noqa: BLE001 - one record's failure must not kill the run
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_battery(
    entries: list[dict], wav_dir: str | Path, deps: EvalDeps, mode: str
) -> list[dict]:
    """Score every manifest entry, resolving each one's audio per ``mode``.

    ``mode == "anchor"`` always scores ``entry["gt_wav"]`` directly - it
    never consults ``wav_dir`` or a generation marker, since ground-truth
    audio always exists and was never "generated". ``mode == "generated"``
    scores ``wav_dir/<example_id>.wav``, gated by the Task 6/7
    ``<example_id>.json`` marker contract: a missing marker, a marker that
    fails to parse as JSON (e.g. truncated by a process killed mid-write,
    before the generator's atomic write-then-``os.replace``), a marker
    carrying an ``"error"``, or ``"has_audio": false`` all short-circuit to
    an errored row without ever touching ``evaluate_record`` (the wav may
    not even exist) - one bad record never aborts the run.
    """
    wav_dir = Path(wav_dir)
    rows = []
    for entry in entries:
        example_id = entry["example_id"]
        if mode == "anchor":
            wav_path = entry["gt_wav"]
        else:
            marker_path = wav_dir / f"{example_id}.json"
            if not marker_path.exists():
                rows.append(
                    _blank_row(example_id, f"missing generation marker: {marker_path}")
                )
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - one bad marker must not abort the run
                rows.append(
                    _blank_row(
                        example_id,
                        f"unreadable generation marker {marker_path}: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if marker.get("error"):
                rows.append(_blank_row(example_id, marker["error"]))
                continue
            if marker.get("has_audio") is False:
                rows.append(
                    _blank_row(example_id, "generation marker reports has_audio=false")
                )
                continue
            wav_path = str(wav_dir / f"{example_id}.wav")
        rows.append(evaluate_record(entry, wav_path, deps, mode=mode))
    return rows


def _pool_counts(counts_dicts: Iterable[dict | None]) -> ErrorCounts | None:
    pooled: ErrorCounts | None = None
    for counts in counts_dicts:
        if counts is None:
            continue
        ec = ErrorCounts(**counts)
        pooled = ec if pooled is None else pooled + ec
    return pooled


def _safe_wer(counts: ErrorCounts | None) -> float | None:
    if counts is None or counts.ref_words == 0:
        return None
    return counts.wer


def _mean_ignore_none(values: Iterable[float | None]) -> float | None:
    # Skip both None (metric not applicable to that row, e.g. a
    # single-cluster window has no distinctness margin) and NaN (a
    # degenerate per-row computation). A single NaN left in would poison
    # the whole mean via ``sum`` - observed suppressing ``sim_margin_mean``
    # to NaN when exactly one of 35 numeric rows was NaN.
    present = [v for v in values if v is not None and not math.isnan(v)]
    return sum(present) / len(present) if present else None


def aggregate(rows: list[dict]) -> dict:
    """Pool ``rows`` into one run-level summary.

    WER metrics pool the I/D/S count dicts across rows (via
    ``ErrorCounts.__add__``) THEN divide once - never averaged per-row.
    ``ErrorCounts.wer`` raises on a zero-ref-word denominator, so pooled
    WER is ``None`` (not a crash) when no row contributed usable counts.
    Sim/UTMOS/purity means and the mapping-disagreement rate all ignore
    rows where the field is ``None`` (Set B rows, errored rows, or
    non-anchor rows for ``purity_gt``).
    """
    wer_concat_counts = _pool_counts(row.get("wer_concat_counts") for row in rows)
    cpwer_counts = _pool_counts(row.get("cpwer_counts") for row in rows)

    disagree_values = (
        (1.0 if row.get("mapping_disagrees") else 0.0)
        if row.get("mapping_disagrees") is not None
        else None
        for row in rows
    )

    return {
        "n_rows": len(rows),
        "n_err": sum(1 for row in rows if row.get("error")),
        "wer_concat": {
            "counts": _counts_to_dict(wer_concat_counts) if wer_concat_counts else None,
            "wer": _safe_wer(wer_concat_counts),
        },
        "cpwer": {
            "counts": _counts_to_dict(cpwer_counts) if cpwer_counts else None,
            "wer": _safe_wer(cpwer_counts),
        },
        "sim_own_mean": _mean_ignore_none(row.get("sim_own_mean") for row in rows),
        "sim_margin_mean": _mean_ignore_none(
            row.get("sim_margin_mean") for row in rows
        ),
        "sim_cross_gt_mean": _mean_ignore_none(
            row.get("sim_cross_gt") for row in rows
        ),
        "utmos_mean": _mean_ignore_none(row.get("utmos") for row in rows),
        "purity_gt_mean": _mean_ignore_none(row.get("purity_gt") for row in rows),
        "mapping_disagreement_rate": _mean_ignore_none(disagree_values),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="path to a manifest JSON")
    ap.add_argument(
        "--wav-dir", required=True, help="directory of generated <id>.wav/.json"
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=("generated", "anchor"),
        help="'generated' scores wav-dir/<id>.wav; 'anchor' scores entry['gt_wav']",
    )
    ap.add_argument("--out", required=True, help="path to write results.json")
    ap.add_argument("--hf-token", default=None, help="pyannote gated-model HF token")
    ap.add_argument("--device", default="cuda")
    return ap


def _build_deps(args: argparse.Namespace) -> EvalDeps:
    return EvalDeps(hf_token=args.hf_token, device=args.device)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    entries = load_manifest(args.manifest)
    deps = _build_deps(args)
    rows = run_battery(entries, args.wav_dir, deps, mode=args.mode)
    agg = aggregate(rows)

    result = {"rows": rows, "aggregate": agg}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"eval done: {agg['n_rows']} rows, {agg['n_err']} errors")
    return 1 if agg["n_rows"] > 0 and agg["n_err"] == agg["n_rows"] else 0


if __name__ == "__main__":
    sys.exit(main())
