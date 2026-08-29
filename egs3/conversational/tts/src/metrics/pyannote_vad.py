"""pyannote voice-activity backend for the Talking Turns judge labels.

The paper (Arora et al., ICLR 2025, Sec. 3.2) derives each side's activity
with the pyannote.audio VAD (Bredin et al., 2020), not Silero, and its
"utterances" are those speech regions. This backend reproduces that: the
``pyannote/voice-activity-detection`` pipeline with its shipped
hyper-parameters (onset 0.81 / offset 0.48, ``min_duration_on`` 0.055 s,
``min_duration_off`` 0.098 s), one pass per channel, returning speech spans
in seconds under the same ``(wav, sr) -> [(start, end)]`` protocol as
:class:`.quality.SileroVADSegmenter`. No dGSLM 200 ms merge is applied on
top: one pyannote segment is one utterance, which is the paper's unit.

``pyannote.audio`` and the gated pipeline weights are imported/fetched on
the first call, never at construction, so this class is safe to build
offline and to put into a config. The Hugging Face token is read the
standard way (``HF_TOKEN`` or ``~/.cache/huggingface/token``) unless
``token`` is passed.

Nothing here is imported by the training path.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

DEFAULT_PIPELINE = "pyannote/voice-activity-detection"


class PyannoteVADSegmenter:
    """Speech spans from a pyannote VAD pipeline.

    ``pipeline`` may be injected (tests, or a pre-loaded object); otherwise
    ``Pipeline.from_pretrained(pipeline_name)`` is loaded lazily and moved
    to ``device``. ``hyper_parameters`` optionally override the pipeline's
    shipped ones via ``instantiate`` (default: none, the paper setting).
    """

    def __init__(
        self,
        pipeline_name: str = DEFAULT_PIPELINE,
        device: str = "cpu",
        token: Optional[str] = None,
        hyper_parameters: Optional[dict] = None,
        pipeline: Any = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.device = device
        self.token = token
        self.hyper_parameters = dict(hyper_parameters or {})
        self._pipeline = pipeline

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from pyannote.audio import Pipeline  # deferred: heavy, gated weights
        except ImportError as exc:
            raise ImportError(
                "PyannoteVADSegmenter requires pyannote.audio (`pip install "
                "pyannote.audio`) and access to the gated "
                f"{self.pipeline_name!r} pipeline (accept its conditions on "
                "huggingface.co and log in)."
            ) from exc
        kwargs = {"use_auth_token": self.token} if self.token else {}
        pipe = Pipeline.from_pretrained(self.pipeline_name, **kwargs)
        if pipe is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained({self.pipeline_name!r}) returned None: "
                "the model is gated - accept its user conditions on "
                "huggingface.co and provide a token (HF_TOKEN or "
                "~/.cache/huggingface/token)."
            )
        if self.hyper_parameters:
            pipe.instantiate(self.hyper_parameters)
        self._pipeline = pipe.to(torch.device(self.device))

    def __call__(self, wav: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        self._load()
        import torch

        waveform = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
        annotation = self._pipeline({"waveform": waveform, "sample_rate": int(sr)})
        return spans_from_annotation(annotation)


def spans_from_annotation(annotation: Any) -> List[Tuple[float, float]]:
    """``[(start, end)]`` in seconds from a pyannote ``Annotation`` (the VAD
    pipeline output): its timeline's support, i.e. overlapping/adjacent
    speech regions merged, sorted by start."""
    timeline = annotation.get_timeline()
    if hasattr(timeline, "support"):
        timeline = timeline.support()
    return sorted((float(seg.start), float(seg.end)) for seg in timeline)
