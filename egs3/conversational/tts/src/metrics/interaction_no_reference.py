"""``InteractionMetric`` variant for runs that have no reference audio.

The shared :class:`~egs3.conversational.tts.src.metrics.interaction
.InteractionMetric` scores each window twice - once over
``channels[ch].gen_wav`` and once over ``channels[ch].gt_wav`` - because its
``{event}_dur_w1`` keys are Wasserstein-1 distances against the ground
truth's event-duration distributions.

The external dialogue test sets (``src/external_inference.py``) ship no
reference audio at all, so their meta JSONs carry no ``gt_wav`` and the
parent would raise ``KeyError`` on every window.  This subclass supplies an
EMPTY ground-truth event set instead, which the parent's own documented
guard (``_wasserstein1(gen, gt) if gen and gt else None``) turns into
``null`` for every ``*_dur_w1`` key.  The count and duration keys
(``{event}_per_min``, ``{event}_sec_per_min``) are computed exactly as in
the parent.

Subclassed rather than merged into the parent on purpose: the SSSD path's
metric code is left byte-identical, so its published numbers remain
reproducible from the same class it was scored with.

READ THESE NUMBERS WITH CARE.  On an audio-free test set the window duration
is PREDICTED, and total duration is the only timing signal the model
receives - a window predicted too long is filled with silence, which moves
pause and gap rates independently of the model.  These keys are here to be
swept against the duration policy (``duration.speed`` in the inference
config), not to be quoted as standalone model properties; turn-taking is
evaluated on the SSSD split, which has real turn times and a ground-truth
anchor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .interaction import EVENT_TYPES, InteractionMetric


class NoReferenceInteractionMetric(InteractionMetric):
    """Interaction battery over generated audio only (no ground truth)."""

    def _score_window(self, meta: Dict[str, Any], test_dir: Path) -> Dict[str, Any]:
        if any("gt_wav" in ch for ch in meta["channels"]):
            raise ValueError(
                f"{meta.get('window_id')}: meta carries gt_wav; use the parent "
                "InteractionMetric so the *_dur_w1 keys are actually computed"
            )
        gen = self._events_for("gen_wav", meta, test_dir)
        empty: Dict[str, List[float]] = {event: [] for event in EVENT_TYPES}
        return {
            "window_id": meta["window_id"],
            "duration_sec": float(meta["window_duration_sec"]),
            "gen_events": gen,
            "gt_events": empty,
        }
