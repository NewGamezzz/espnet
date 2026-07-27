"""Speaker diarization wrapper (Task 4) and diarization purity metric.

`diarize()` wraps `pyannote.audio.Pipeline` - the heavy, GPU-bound model
import happens lazily, inside the function only, so importing this module
never pulls in `pyannote`. The pipeline is cached in a module global
because loading it is expensive and Task 8 calls `diarize()` once per
eval window.

`purity()` is PURE logic (no audio, no models): a time-weighted fraction
of diarized speech that a majority-overlap-consistent speaker assignment
would explain, matching the "BagPiper Conversational Baseline Eval"
plan's diarization-quality metric. Task 8 calls it with GT turns straight
from the eval manifest (window-relative seconds).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_pipeline = None


@dataclass
class DiarSegment:
    start: float
    end: float
    cluster: str


def diarize(
    wav_path: str,
    hf_token: str | None = None,
    device: str = "cuda",
    num_speakers: int | None = None,
) -> list[DiarSegment]:
    """Run pyannote speaker diarization on `wav_path`, returning segments
    sorted by start time.

    `num_speakers` forces the pipeline to exactly that many clusters;
    None (the default) leaves the speaker count unconstrained.

    Lazily imports `pyannote.audio.Pipeline` and `torch` (never at module
    scope) and loads `pyannote/speaker-diarization-3.1` (a gated model -
    `hf_token`, falling back to the `HF_TOKEN` env var, authenticates the
    download). The loaded pipeline is cached in a module global so repeat
    calls across an eval run pay the load cost once.
    """
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        # pyannote/speaker-diarization-3.1 ships a PyTorch-Lightning
        # checkpoint whose `torch.load` pulls globals (e.g.
        # `torch.torch_version.TorchVersion`) that are not on PyTorch
        # 2.6's `weights_only=True` allowlist, so the load raises
        # UnpicklingError under torch >= 2.6. The model is a gated,
        # explicitly license-accepted download (a trusted source), so we
        # force `weights_only=False` for the duration of the load only,
        # then restore `torch.load`. This is narrower and more robust
        # than allowlisting the checkpoint's globals one at a time.
        _orig_load = torch.load

        def _trusting_load(*args, **kwargs):
            # Force (not setdefault): pyannote loads its checkpoint via
            # PyTorch-Lightning's `pl_load`, which passes
            # `weights_only=True` explicitly, so a setdefault would be a
            # no-op. Overriding is safe here - the model is a trusted,
            # license-accepted download and the override is scoped to
            # this one load.
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _trusting_load
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token or os.environ.get("HF_TOKEN"),
            )
        finally:
            torch.load = _orig_load
        _pipeline = pipeline.to(torch.device(device))

    kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
    annotation = _pipeline(wav_path, **kwargs)
    segments = [
        DiarSegment(start=segment.start, end=segment.end, cluster=label)
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(segments, key=lambda s: s.start)


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def purity(segments: list[DiarSegment], gt_turns: list[dict]) -> float:
    """Time-weighted diarization purity against ground-truth turns.

    For each diarized cluster, sums its overlap time with each GT speaker
    (across every segment in that cluster and every GT turn), then takes
    the max over speakers - the overlap time a "majority speaker" label
    for that cluster would correctly explain. The result is that summed
    max, divided by the total segment-GT overlap time across all clusters
    and speakers. Segment time that overlaps no GT turn contributes to
    neither the numerator nor the denominator. Returns 0.0 when there is
    no overlap at all (rather than dividing by zero).
    """
    cluster_speaker_overlap: dict[str, dict[str, float]] = {}
    for segment in segments:
        for turn in gt_turns:
            ov = _overlap(segment.start, segment.end, turn["start"], turn["end"])
            if ov <= 0:
                continue
            speaker_overlap = cluster_speaker_overlap.setdefault(segment.cluster, {})
            speaker_overlap[turn["speaker"]] = (
                speaker_overlap.get(turn["speaker"], 0.0) + ov
            )

    total_overlap = sum(
        ov
        for speaker_overlap in cluster_speaker_overlap.values()
        for ov in speaker_overlap.values()
    )
    if total_overlap <= 0:
        return 0.0

    correct_overlap = sum(
        max(speaker_overlap.values())
        for speaker_overlap in cluster_speaker_overlap.values()
    )
    return correct_overlap / total_overlap
