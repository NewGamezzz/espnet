"""``InteractionMetric``: the dGSLM turn-taking battery (arXiv 2203.16502),
the headline metric family of the conversational eval - it is what
distinguishes a conversational model from N parallel TTS systems, and it is
exactly what the zero-gate pretrained baseline is expected to fail.

Event definitions (dGSLM, computed per window from per-channel Silero VAD
IPUs -- speech spans bounded by >= 200 ms of silence, the same
:class:`~.quality.SileroVADSegmenter` unit PR #12 ships):

* **IPU** -- one VAD speech span of one channel.
* **Overlap** -- a maximal interval where BOTH channels have speech.
* **Silence** -- a maximal interval where NO channel has speech. A silence
  is a **pause** if the same channel holds the floor on both sides (the
  last IPU to end before it and the first IPU to start after it belong to
  the same channel), else a **gap** (the floor changed hands). Window-edge
  silences (no speech before, or none after) are neither: they have no
  before/after speaker, and are skipped -- matching dGSLM, which measures
  transitions, not endpoints.

Reported per event type ``e`` in ``{ipu, pause, gap, overlap}`` (all
``summary_value``-guarded: undefined stays ``None``, never a fabricated
0.0):

* ``{e}_per_min`` -- events per minute, pooled: total event count across
  the run divided by total window minutes (never a mean of per-window
  rates, mirroring the corpus-WER pooling rule).
* ``{e}_sec_per_min`` -- cumulated event seconds per window minute, pooled
  the same way (dGSLM's "cumulated duration per minute").
* ``{e}_dur_w1`` -- Wasserstein-1 distance (seconds) between the
  event-duration distribution of the GENERATED audio and that of the
  paired GROUND TRUTH, both pooled over the run. The reference needs no
  cross-directory pairing: every window's meta already carries
  ``channels[ch].gt_wav``, so the ground-truth events come from the same
  infer dir being scored. In ``gt`` mode ``gen_wav`` IS the ground truth,
  so every ``*_dur_w1`` collapses to ~0.0 -- a built-in sanity check, and
  the reason anchors need no metric-side special-casing here either.
  ``None`` when either side of a distribution is empty (W1 against
  nothing is undefined, not 0).

Interpreting the numbers: read generated rates against the ``gt`` anchor
run's rates (same windows, same VAD), and W1 directly as "how far the
model's event-duration distributions are from ground truth". The zero-gate
pretrained signature is heavy unstructured overlap, few gaps, and large W1
everywhere; fine-tuning should pull all twelve numbers toward the anchor.

Backends: ``vad_backend`` is constructor-injectable with the same
:class:`~.quality.SileroVADSegmenter` default as ``QualityMetric`` (lazy
faster-whisper import, no new dependency). W1 uses
``scipy.stats.wasserstein_distance`` (scipy is already an espnet
dependency), soft-imported at first use.

Deferred (see README's deferred list): backchannel proxy
(short-IPU-during-overlap), laughter statistics, and the Fisher reference
corpus for cross-corpus W1.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import load_wav, summary_value
from .quality import SileroVADSegmenter, VADBackend

Span = Tuple[float, float]

EVENT_TYPES = ("ipu", "pause", "gap", "overlap")


# --------------------------------------------------------------------------- #
# pure interval arithmetic (unit-testable without any VAD)
# --------------------------------------------------------------------------- #
def _merge_spans(spans: Sequence[Span]) -> List[Span]:
    """Sort and merge overlapping/adjacent spans into disjoint spans."""
    merged: List[Span] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect(a: Sequence[Span], b: Sequence[Span]) -> List[Span]:
    """Pairwise intersections of two disjoint, sorted span lists."""
    out: List[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _complement(spans: Sequence[Span], total_sec: float) -> List[Span]:
    """Maximal intervals of [0, total_sec] not covered by ``spans``
    (which must be disjoint and sorted)."""
    out: List[Span] = []
    cursor = 0.0
    for start, end in spans:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if total_sec > cursor:
        out.append((cursor, total_sec))
    return out


def derive_events(
    channel_ipus: Sequence[Sequence[Span]], total_sec: float
) -> Dict[str, List[float]]:
    """dGSLM event durations from per-channel IPU spans.

    Returns ``{event_type: [duration_sec, ...]}`` for the four event types.
    Window-edge silences are skipped (no before/after speaker). Pause vs
    gap attribution uses the channel of the last IPU to END before the
    silence and the first IPU to START after it.
    """
    per_channel = [_merge_spans(ch) for ch in channel_ipus]

    ipus = [end - start for ch in per_channel for start, end in ch]

    overlaps: List[Span] = []
    for i in range(len(per_channel)):
        for j in range(i + 1, len(per_channel)):
            overlaps.extend(_intersect(per_channel[i], per_channel[j]))
    overlaps = _merge_spans(overlaps)

    any_speech = _merge_spans([s for ch in per_channel for s in ch])
    silences = _complement(any_speech, total_sec)

    pauses: List[float] = []
    gaps: List[float] = []
    for start, end in silences:
        before = _last_channel_ending_by(per_channel, start)
        after = _first_channel_starting_from(per_channel, end)
        if before is None or after is None:
            continue  # window-edge silence: no transition to classify
        (pauses if before == after else gaps).append(end - start)

    return {
        "ipu": ipus,
        "pause": pauses,
        "gap": gaps,
        "overlap": [end - start for start, end in overlaps],
    }


def _last_channel_ending_by(
    per_channel: Sequence[Sequence[Span]], t: float
) -> Optional[int]:
    """Channel of the IPU with the latest end <= t (None if no IPU ends
    by ``t``)."""
    best_end, best_ch = None, None
    for ch, spans in enumerate(per_channel):
        for _start, end in spans:
            if end <= t and (best_end is None or end > best_end):
                best_end, best_ch = end, ch
    return best_ch


def _first_channel_starting_from(
    per_channel: Sequence[Sequence[Span]], t: float
) -> Optional[int]:
    """Channel of the IPU with the earliest start >= t (None if no IPU
    starts at or after ``t``)."""
    best_start, best_ch = None, None
    for ch, spans in enumerate(per_channel):
        for start, _end in spans:
            if start >= t and (best_start is None or start < best_start):
                best_start, best_ch = start, ch
    return best_ch


def _wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import wasserstein_distance  # soft import, first use

    return float(wasserstein_distance(a, b))


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class InteractionMetric(BaseMetric):
    """dGSLM turn-taking battery: per-minute event rates and cumulated
    durations for the generated audio, plus W1 duration-distribution
    distances against the paired ground truth. See module docstring."""

    def __init__(
        self,
        vad_backend: Optional[VADBackend] = None,
        vad_sample_rate: int = 16000,
    ) -> None:
        self.vad_backend = (
            vad_backend if vad_backend is not None else SileroVADSegmenter()
        )
        self.vad_sample_rate = vad_sample_rate

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "interaction"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_window: List[Dict[str, Any]] = []
        with (out_dir / "windows.jsonl").open("w", encoding="utf-8") as fout:
            for _window_id, row in self.iter_inputs(data, "meta"):
                meta = json.loads((test_dir / row["meta"]).read_text("utf-8"))
                record = self._score_window(meta, test_dir)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                per_window.append(record)

        summary = self._summarize(per_window)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary

    # -- per-window scoring ------------------------------------------------ #
    def _events_for(
        self, wav_key: str, meta: Dict[str, Any], test_dir: Path
    ) -> Dict[str, List[float]]:
        channel_ipus = []
        for ch in meta["channels"]:
            wav, sr = load_wav(test_dir / ch[wav_key], target_sr=self.vad_sample_rate)
            channel_ipus.append(self.vad_backend(wav, sr))
        return derive_events(channel_ipus, float(meta["window_duration_sec"]))

    def _score_window(self, meta: Dict[str, Any], test_dir: Path) -> Dict[str, Any]:
        gen = self._events_for("gen_wav", meta, test_dir)
        gt = self._events_for("gt_wav", meta, test_dir)
        return {
            "window_id": meta["window_id"],
            "duration_sec": float(meta["window_duration_sec"]),
            "gen_events": gen,
            "gt_events": gt,
        }

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        total_min = sum(w["duration_sec"] for w in per_window) / 60.0

        summary: Dict[str, Optional[float]] = {}
        for event in EVENT_TYPES:
            gen = [d for w in per_window for d in w["gen_events"][event]]
            gt = [d for w in per_window for d in w["gt_events"][event]]
            summary[f"{event}_per_min"] = summary_value(
                len(gen) / total_min if total_min > 0 else None,
                f"{event}_per_min",
                metric_name=name,
            )
            summary[f"{event}_sec_per_min"] = summary_value(
                sum(gen) / total_min if total_min > 0 else None,
                f"{event}_sec_per_min",
                metric_name=name,
            )
            summary[f"{event}_dur_w1"] = summary_value(
                _wasserstein1(gen, gt) if gen and gt else None,
                f"{event}_dur_w1",
                metric_name=name,
            )
        return summary
