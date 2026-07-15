"""``InteractionMetric``: the dGSLM-style turn-taking leg of the measure-stage
metric battery. VAD only -- no ASR.

Per window (iterating ``meta.scp``), for every channel: segment BOTH the
GENERATED region (``gen_wav``) and its paired GROUND-TRUTH region (``gt_wav``,
"same length domain as gen" per the infer stage's output contract) into IPUs
via ``segments.py``'s dGSLM 200 ms rule (``build_ipus``). The rest of this
module is pure interval algebra over those per-channel IPU lists -- no audio
touches the math past this point.

Event definitions (operationalized exactly as follows; N is the channel
count, not hardcoded to 2):

* **IPU** -- one channel's own build_ipus output; the per-window IPU
  population is the POOLED list across all channels (their union, not a
  per-channel figure).
* **Overlap** -- maximal intervals where >= 2 channels have simultaneous
  speech. Computed with a sweep over every channel-IPU boundary instant: the
  window is cut into segments by every IPU start/end across every channel,
  each segment's "active channel count" is evaluated once (at its midpoint,
  since IPUs from build_ipus are already non-overlapping *within* a channel
  so the count is constant across a segment), and segments with count >= 2
  are merged into overlap intervals (:func:`_active_count_segments`,
  :func:`_merge_predicate_intervals`).
* **Silence intervals** -- maximal intervals from the SAME sweep where NO
  channel is active (count == 0), restricted to ``[0, region_duration]``.
  Each silence interval bounded by speech on both sides is classified by
  comparing the set of channels whose IPU ends EXACTLY at the silence's
  start ("preceding") against the set whose IPU starts EXACTLY at the
  silence's end ("following"): a **gap** if these sets are DISJOINT (a
  speaker switch happened), a **pause** if they INTERSECT (at least one
  channel that was active right before is also active right after, i.e. the
  same speaker resumes). Using sets rather than a single "the" preceding/
  following channel generalizes the task's N=2 "differs vs. same" rule to
  simultaneous multi-channel stops/starts (a real but rare tie in synthetic
  fixtures); the generalization is exercised explicitly by
  ``TestClassifySilence::test_tied_preceding_channels_intersecting_with_a_single_following_channel_is_a_pause``.
  A silence interval touching the region's start or end (``start <= 0`` or
  ``end >= region_duration``) is LEADING/TRAILING and counted as NEITHER gap
  nor pause (including the degenerate all-silence window, which is both at
  once).
* **Backchannel proxy** -- a short-IPU-during-overlap rate, computed on the
  GENERATED condition only (the plan does not ask for a paired GT figure
  here, unlike the four event types above): an IPU shorter than
  ``backchannel_max_sec`` (default 1.0 s) whose duration overlaps ANOTHER
  channel's IPUs by a fraction >= ``backchannel_overlap_frac`` (default 0.5,
  i.e. "mostly inside another channel's speech") counts as one backchannel
  event; ``backchannel_per_min`` is that count normalized by the region
  duration. This is INDEPENDENT of (and typically overlaps with) the
  "overlap" event type above: overlap measures simultaneous-speech TIME,
  backchannel counts short IPUs that happen to sit mostly inside another
  channel's speech -- a short IPU can (and typically does) drive both an
  overlap interval and a backchannel count simultaneously; they are not
  mutually exclusive and are not meant to be.

Per window, for each event type in {ipu, pause, gap, overlap} on the
GENERATED condition: events per minute and cumulated-duration per minute,
BOTH normalized by the generated region's own duration in minutes
(``(window_duration_sec - prompt_boundary_sec) / 60``, identical for the
paired GT audio per the "same length domain" contract). Additionally, for
each event type, the Wasserstein-1 distance (``scipy.stats.
wasserstein_distance``) between that window's GENERATED and GROUND-TRUTH
event-DURATION lists (each list's raw per-event durations, unweighted point
masses -- scipy's default). Convention when either side has zero events for
that type in that window: the window is SKIPPED for that w1 key (not
fabricated as 0), and the summary key is a mean over the windows that DID
have both sides populated, using the shared aggregation helpers in
``_common.py`` (``mean_skip_none`` / ``summary_value``, same as
``ConversationASRMetric`` and ``SpeakerDynamicsMetric``). This makes
``w1_*`` a per-window scalar
meaned over windows -- UNLIKE ``SpeakerDynamicsMetric``'s ``bleed_db_p50/90``,
which pools raw pairs across the whole run; there is no cross-window pooling
here because the plan's "Per window: ... Wasserstein-1 distance ..." framing
places the computation at window granularity.

GT-side per-event-type stats (counts, durations) are written to the
per-window JSONL as diagnostics; they are NEVER read into the summary
computation. Each per-window JSONL record also carries ``w1_skipped``, the
list of event-type names skipped for W1 in THAT window (mirroring
``SpeakerDynamicsMetric``'s ``bleed_skipped_pairs`` -- the plan's "skip +
count" convention: the skip is a ``None`` value, and the count is this
explicit list, not merely inferable from an empty duration list).

Summary keys (13 floats, or 15 when laughter is enabled): ``ipu_per_min``,
``pause_per_min``, ``gap_per_min``, ``overlap_per_min``, ``ipu_dur_per_min``,
``pause_dur_per_min``, ``gap_dur_per_min``, ``overlap_dur_per_min``,
``w1_ipu``, ``w1_pause``, ``w1_gap``, ``w1_overlap``, ``backchannel_per_min``
(+ ``laughter_per_min``, ``laughter_mean_dur`` when a laughter detector is
injected). Each is a mean over windows of the per-window value; a summary
key with zero defined values anywhere is left ``None`` (serializes as JSON
``null``, rendered as ``-`` by ``local/eval_report.py``) with a logged
warning, rather than a fabricated 0.0 (same convention as the other metric
classes; see ``_common.py``).

Laughter (plan: optional, timeboxed, gated) is OFF by default
(``laughter_detector=None``): when disabled, no laughter keys appear
anywhere (JSONL or summary) and nothing laughter-related is computed. When a
detector is injected -- any ``(wav, sr) -> Sequence[(start_sec, end_sec)]``
callable, mirroring the ``VADBackend`` interface -- it is called once per
GENERATED channel wav per window; ``laughter_per_min`` is the pooled
event-count rate (same per-minute normalization as the other event types)
and ``laughter_mean_dur`` is the mean duration of laughter events IN THAT
WINDOW, then both are meaned over windows for the summary. This module does
NOT vendor or download any laughter research code: the real detector
(candidate: ``jrgillick/laughter-detection``, an unpackaged research repo,
not a pip-installable dependency) is intentionally left un-wired here and is
a task for first use on Delta, where it can be cloned/adapted once and
tested against real audio; a fake detector (a fixed list of intervals)
exercises the enabled code path in this module's tests.

Backends are constructor-injectable; the real default (``vad``) is lazy so
constructing this metric (e.g. from ``conf/metrics.yaml`` offline) never
touches the network or loads a model:

* ``vad``: default ``segments.VAD()`` (lazy silero), per the shared Task-2
  utility, used for BOTH the generated and ground-truth IPU segmentation.
* ``laughter_detector``: no real default; ``None`` disables the feature
  entirely (see above).

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import mean, mean_skip_none, summary_value
from .segments import VAD, Interval, build_ipus, load_wav

_EPS = 1e-6

LaughterBackend = Callable[[np.ndarray, int], Sequence[Interval]]


# --------------------------------------------------------------------------- #
# pure interval algebra: sweep-line over per-channel IPU lists
# --------------------------------------------------------------------------- #
def _boundary_times(
    channel_ipus: Sequence[Sequence[Interval]], region_duration: float
) -> List[float]:
    times = {0.0, float(region_duration)}
    for ipus in channel_ipus:
        for s, e in ipus:
            times.add(round(float(s), 9))
            times.add(round(float(e), 9))
    return sorted(t for t in times if -_EPS <= t <= region_duration + _EPS)


def _active_count_segments(
    channel_ipus: Sequence[Sequence[Interval]], region_duration: float
) -> List[Tuple[float, float, int]]:
    """Partition ``[0, region_duration]`` at every channel-IPU boundary and
    tag each resulting segment with how many channels are active on it
    (evaluated at the segment's midpoint, valid since a single channel's
    IPUs never overlap each other)."""
    times = _boundary_times(channel_ipus, region_duration)
    segments: List[Tuple[float, float, int]] = []
    for t0, t1 in zip(times, times[1:]):
        if t1 - t0 <= _EPS:
            continue
        mid = (t0 + t1) / 2.0
        count = sum(
            1
            for ipus in channel_ipus
            for s, e in ipus
            if s - _EPS <= mid <= e + _EPS
        )
        segments.append((t0, t1, count))
    return segments


def _merge_predicate_intervals(
    segments: Sequence[Tuple[float, float, int]],
    predicate: Callable[[int], bool],
) -> List[Interval]:
    """Merge consecutive segments satisfying ``predicate(count)`` into
    intervals. Segments are contiguous and already sorted by construction, so
    two accepted segments are adjacent iff nothing was rejected between
    them."""
    out: List[Interval] = []
    for s, e, count in segments:
        if not predicate(count):
            continue
        if out and s <= out[-1][1] + _EPS:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _classify_silence(
    channel_ipus: Sequence[Sequence[Interval]],
    start: float,
    end: float,
    region_duration: float,
) -> Optional[str]:
    """``"gap"``, ``"pause"``, or ``None`` (leading/trailing/whole-window
    silence) -- see module docstring for the set-based generalization."""
    if start <= _EPS or end >= region_duration - _EPS:
        return None
    preceding = {
        ch
        for ch, ipus in enumerate(channel_ipus)
        for _s, e in ipus
        if abs(e - start) <= _EPS
    }
    following = {
        ch
        for ch, ipus in enumerate(channel_ipus)
        for s, _e in ipus
        if abs(s - end) <= _EPS
    }
    return "pause" if (preceding & following) else "gap"


@dataclass
class EventBattery:
    """One condition's (generated or ground-truth) pooled event durations."""

    ipu_durations: List[float] = field(default_factory=list)
    pause_durations: List[float] = field(default_factory=list)
    gap_durations: List[float] = field(default_factory=list)
    overlap_durations: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "ipu_durations": self.ipu_durations,
            "pause_durations": self.pause_durations,
            "gap_durations": self.gap_durations,
            "overlap_durations": self.overlap_durations,
        }


def _compute_event_battery(
    channel_ipus: Sequence[Sequence[Interval]], region_duration: float
) -> EventBattery:
    ipu_durations = [e - s for ipus in channel_ipus for s, e in ipus]
    segments = _active_count_segments(channel_ipus, region_duration)
    overlap_intervals = _merge_predicate_intervals(segments, lambda c: c >= 2)
    silence_intervals = _merge_predicate_intervals(segments, lambda c: c == 0)

    pause_durations: List[float] = []
    gap_durations: List[float] = []
    for s, e in silence_intervals:
        kind = _classify_silence(channel_ipus, s, e, region_duration)
        if kind == "pause":
            pause_durations.append(e - s)
        elif kind == "gap":
            gap_durations.append(e - s)

    overlap_durations = [e - s for s, e in overlap_intervals]
    return EventBattery(
        ipu_durations, pause_durations, gap_durations, overlap_durations
    )


def _rate_stats(
    durations: Sequence[float], duration_minutes: float
) -> Tuple[Optional[float], Optional[float]]:
    """``(events_per_min, cumulated_duration_per_min)``; ``None, None`` when
    the region has no measurable duration."""
    if duration_minutes <= _EPS:
        return None, None
    return len(durations) / duration_minutes, sum(durations) / duration_minutes


def _w1(
    gen_durations: Sequence[float], gt_durations: Sequence[float]
) -> Optional[float]:
    """Wasserstein-1 distance between two event-duration distributions;
    ``None`` (skip, never a fabricated 0) when either side is empty."""
    if not gen_durations or not gt_durations:
        return None
    return float(wasserstein_distance(list(gen_durations), list(gt_durations)))


def _overlap_with_other_channels(
    ipu: Interval, own_channel: int, channel_ipus: Sequence[Sequence[Interval]]
) -> float:
    s0, e0 = ipu
    total = 0.0
    for ch, ipus in enumerate(channel_ipus):
        if ch == own_channel:
            continue
        for s, e in ipus:
            total += max(0.0, min(e0, e) - max(s0, s))
    return total


def _count_backchannels(
    channel_ipus: Sequence[Sequence[Interval]],
    backchannel_max_sec: float,
    backchannel_overlap_frac: float,
) -> int:
    count = 0
    for ch, ipus in enumerate(channel_ipus):
        for s, e in ipus:
            dur = e - s
            if dur <= _EPS or dur >= backchannel_max_sec:
                continue
            overlap = _overlap_with_other_channels((s, e), ch, channel_ipus)
            if overlap / dur >= backchannel_overlap_frac - _EPS:
                count += 1
    return count


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
_EVENT_TYPES = ("ipu", "pause", "gap", "overlap")


class InteractionMetric(BaseMetric):
    """dGSLM-style turn-taking leg of the measure-stage battery: IPU/pause/
    gap/overlap rates, per-event-type Wasserstein-1 vs. ground truth, and a
    short-IPU-during-overlap backchannel proxy. See module docstring."""

    def __init__(
        self,
        vad: Optional[Callable[[np.ndarray, int], Sequence[Interval]]] = None,
        laughter_detector: Optional[LaughterBackend] = None,
        interaction_sample_rate: int = 16000,
        min_silence: float = 0.2,
        min_speech: float = 0.0,
        pad: float = 0.0,
        backchannel_max_sec: float = 1.0,
        backchannel_overlap_frac: float = 0.5,
    ) -> None:
        self.vad = vad if vad is not None else VAD()
        self.laughter_detector = laughter_detector
        self.interaction_sample_rate = interaction_sample_rate
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.pad = pad
        self.backchannel_max_sec = backchannel_max_sec
        self.backchannel_overlap_frac = backchannel_overlap_frac

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
    def _ipus_for(self, wav_path: Path, region_duration: float):
        wav, sr = load_wav(wav_path, target_sr=self.interaction_sample_rate)
        raw = self.vad(wav, sr)
        ipus = build_ipus(
            raw,
            min_silence=self.min_silence,
            min_speech=self.min_speech,
            pad=self.pad,
            total_duration=region_duration,
        )
        return wav, sr, ipus

    def _score_window(self, meta: Dict[str, Any], test_dir: Path) -> Dict[str, Any]:
        window_id = meta["window_id"]
        channels_meta = meta["channels"]
        n = len(channels_meta)
        region_duration = float(meta["window_duration_sec"]) - float(
            meta["prompt_boundary_sec"]
        )
        duration_minutes = region_duration / 60.0

        gen_wavs: List[Tuple[np.ndarray, int]] = []
        gen_ipus: List[List[Interval]] = []
        gt_ipus: List[List[Interval]] = []
        for ch in channels_meta:
            wav, sr, ipus = self._ipus_for(test_dir / ch["gen_wav"], region_duration)
            gen_wavs.append((wav, sr))
            gen_ipus.append(ipus)
            _gt_wav, _gt_sr, gt_ipu_list = self._ipus_for(
                test_dir / ch["gt_wav"], region_duration
            )
            gt_ipus.append(gt_ipu_list)

        gen_battery = _compute_event_battery(gen_ipus, region_duration)
        gt_battery = _compute_event_battery(gt_ipus, region_duration)

        record: Dict[str, Any] = {
            "window_id": window_id,
            "num_channels": n,
            "region_duration_sec": region_duration,
            "generated": gen_battery.as_dict(),
            "ground_truth": gt_battery.as_dict(),
        }

        w1_skipped: List[str] = []
        for event_type in _EVENT_TYPES:
            gen_durs = getattr(gen_battery, f"{event_type}_durations")
            gt_durs = getattr(gt_battery, f"{event_type}_durations")
            rate, dur_rate = _rate_stats(gen_durs, duration_minutes)
            record[f"{event_type}_per_min"] = rate
            record[f"{event_type}_dur_per_min"] = dur_rate
            w1 = _w1(gen_durs, gt_durs)
            record[f"w1_{event_type}"] = w1
            if w1 is None:
                w1_skipped.append(event_type)
        record["w1_skipped"] = w1_skipped

        backchannel_count = _count_backchannels(
            gen_ipus, self.backchannel_max_sec, self.backchannel_overlap_frac
        )
        record["backchannel_count"] = backchannel_count
        record["backchannel_per_min"] = (
            backchannel_count / duration_minutes
            if duration_minutes > _EPS
            else None
        )

        if self.laughter_detector is not None:
            laughter_events: List[Interval] = []
            for wav, sr in gen_wavs:
                laughter_events.extend(self.laughter_detector(wav, sr))
            laughter_durations = [float(e) - float(s) for s, e in laughter_events]
            record["laughter_count"] = len(laughter_events)
            record["laughter_per_min"] = (
                len(laughter_events) / duration_minutes
                if duration_minutes > _EPS
                else None
            )
            record["laughter_mean_dur"] = mean(laughter_durations)

        return record

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        def agg(key: str) -> Optional[float]:
            return mean_skip_none(w[key] for w in per_window)

        name = type(self).__name__
        summary: Dict[str, Optional[float]] = {}
        for event_type in _EVENT_TYPES:
            summary[f"{event_type}_per_min"] = summary_value(
                agg(f"{event_type}_per_min"), f"{event_type}_per_min", metric_name=name
            )
            summary[f"{event_type}_dur_per_min"] = summary_value(
                agg(f"{event_type}_dur_per_min"),
                f"{event_type}_dur_per_min",
                metric_name=name,
            )
            summary[f"w1_{event_type}"] = summary_value(
                agg(f"w1_{event_type}"), f"w1_{event_type}", metric_name=name
            )
        summary["backchannel_per_min"] = summary_value(
            agg("backchannel_per_min"), "backchannel_per_min", metric_name=name
        )

        if self.laughter_detector is not None:
            summary["laughter_per_min"] = summary_value(
                agg("laughter_per_min"), "laughter_per_min", metric_name=name
            )
            summary["laughter_mean_dur"] = summary_value(
                agg("laughter_mean_dur"), "laughter_mean_dur", metric_name=name
            )

        return summary
