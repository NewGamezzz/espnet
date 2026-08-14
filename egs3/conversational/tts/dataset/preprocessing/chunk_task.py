"""Special-token conditioning chunk-task plan: dataclasses and JSON codec.

Pure module: no I/O, no torch.  A ``ChunkTaskPlan`` records what an online
window sampler decided to do for special-token conditioning training - all
spans are session-absolute seconds (same coordinate space as ``Turn.start``/
``Turn.end`` and ``WindowRecord.t0``/``t1``), never window-relative.  This
module must never import ``windows.py``: ``windows.py`` imports FROM here
(``WindowRecord`` carries an optional ``ChunkTaskPlan``), and a reverse import
would create a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

_KINDS = frozenset({"full", "prompt_only"})


@dataclass(frozen=True)
class ChunkTaskParams:
    """Sampling knobs for the special-token chunk task (values TBD by the
    training design; defaults here are placeholders, not ratified live
    settings)."""

    chunk_task_prob: float = 0.0
    prompt_only_prob: float = 0.2
    chunk_window_min: float = 15.0
    chunk_window_max: float = 35.0
    prev_slice_min: float = 2.0
    prev_slice_max: float = 10.0
    prompt_slice_min: float = 3.0
    prompt_slice_max: float = 8.0
    prompt_speech_floor: float = 3.0


@dataclass(frozen=True)
class ChunkTaskPlan:
    """One sampled chunk-task decision for a window.

    ``kind`` selects between a "full" plan (a <prev_chunk> slice precedes the
    window) and a "prompt_only" plan (no previous-chunk conditioning).
    ``prev_span`` is session-absolute and present iff ``kind == "full"``.
    ``prompt_spans`` has exactly one span per ORIGINAL channel (i.e. every
    speaker gets a voice reference - see the "prompt must cover all speakers"
    rule), all of equal length so the prompt batches as a fixed-size tensor.
    """

    kind: str  # "full" | "prompt_only"
    prev_span: tuple[float, float] | None  # session-absolute; None iff prompt_only
    prompt_spans: tuple[
        tuple[float, float], ...
    ]  # one per ORIGINAL channel, equal lengths

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        has_prev = self.prev_span is not None
        if has_prev != (self.kind == "full"):
            raise ValueError(
                "prev_span is None iff kind == 'prompt_only': got "
                f"kind={self.kind!r}, prev_span={self.prev_span!r}"
            )
        if has_prev and not (self.prev_span[0] < self.prev_span[1]):
            raise ValueError(f"prev_span must have start < end, got {self.prev_span!r}")
        if not self.prompt_spans:
            raise ValueError("prompt_spans must be non-empty")
        lengths = set()
        for start, end in self.prompt_spans:
            if not (start < end):
                raise ValueError(
                    f"prompt span must have start < end, got {(start, end)!r}"
                )
            lengths.add(round(end - start, 6))
        if len(lengths) != 1:
            raise ValueError(
                f"all prompt_spans must have equal (rounded) length, got lengths {lengths}"
            )


def plan_to_json(p: ChunkTaskPlan) -> dict:
    return {
        "kind": p.kind,
        "prev_span": list(p.prev_span) if p.prev_span is not None else None,
        "prompt_spans": [list(span) for span in p.prompt_spans],
    }


def plan_from_json(d: dict) -> ChunkTaskPlan:
    prev_span = d["prev_span"]
    return ChunkTaskPlan(
        kind=d["kind"],
        prev_span=tuple(prev_span) if prev_span is not None else None,
        prompt_spans=tuple(tuple(span) for span in d["prompt_spans"]),
    )
