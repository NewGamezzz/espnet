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
        silence_gate_dbfs: Optional[float] = None,
        gate_frame_ms: float = 20.0,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.device = device
        self.token = token
        self.hyper_parameters = dict(hyper_parameters or {})
        self._pipeline = pipeline
        # Optional absolute silence gate (see :func:`silence_gate`). The
        # pyannote segmentation model is amplitude-normalised (SincNet +
        # instance norm) and reads a faint speech-shaped vocoder residue on
        # a synthetic system's idle channel (-79 dBFS measured on the
        # ZipVoice-Dialog stereo baseline) as speech 64 % of the time. The
        # paper's human recordings never had such near-digital silence.
        # Applied identically to every system, anchor included; None = off.
        self.silence_gate_dbfs = silence_gate_dbfs
        self.gate_frame_ms = gate_frame_ms

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch

            _shim_pyannote_compat()
            from pyannote.audio import Pipeline  # deferred: heavy, gated weights
        except ImportError as exc:
            raise ImportError(
                "PyannoteVADSegmenter requires pyannote.audio (`pip install "
                "pyannote.audio`) and access to the gated "
                f"{self.pipeline_name!r} pipeline (accept its conditions on "
                "huggingface.co and log in)."
            ) from exc
        kwargs = {"use_auth_token": self.token} if self.token else {}
        with _torch_load_full():
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

        wav = np.asarray(wav, dtype=np.float32)
        if self.silence_gate_dbfs is not None:
            wav = silence_gate(wav, sr, self.silence_gate_dbfs, self.gate_frame_ms)
        waveform = torch.as_tensor(wav).reshape(1, -1)
        annotation = self._pipeline({"waveform": waveform, "sample_rate": int(sr)})
        return spans_from_annotation(annotation)


def _shim_pyannote_compat() -> None:
    """pyannote.audio 3.3.2 (the paper-era release that still ships the
    ``pyannote/voice-activity-detection`` pipeline) predates huggingface_hub
    1.x, which dropped the ``use_auth_token`` keyword it passes to
    ``hf_hub_download``. Wrap the function IN PLACE (idempotent) before
    pyannote imports it, translating the keyword; a hub that still accepts
    it is left alone."""
    import inspect

    import huggingface_hub

    fn = huggingface_hub.hf_hub_download
    if getattr(fn, "_pyannote_shim", False):
        return
    if "use_auth_token" in inspect.signature(fn).parameters:
        return

    def hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return fn(*args, **kwargs)

    hf_hub_download._pyannote_shim = True  # type: ignore[attr-defined]
    huggingface_hub.hf_hub_download = hf_hub_download


class _torch_load_full:
    """torch >= 2.6 defaults ``torch.load(weights_only=True)``, which refuses
    the pickled hyper-parameter objects inside pyannote 3.x checkpoints.
    Restore the pre-2.6 default only while the pipeline is being loaded;
    the weights are the gated, licence-accepted Hugging Face files."""

    def __enter__(self):
        import torch

        self._orig = torch.load

        def load(*a, **k):
            # lightning's pl_load passes weights_only=True explicitly on
            # torch >= 2.6, so this must override, not default
            k["weights_only"] = False
            return self._orig(*a, **k)

        torch.load = load
        return self

    def __exit__(self, *exc):
        import torch

        torch.load = self._orig
        return False


def silence_gate(
    wav: np.ndarray, sr: int, threshold_dbfs: float, frame_ms: float = 20.0
) -> np.ndarray:
    """Zero every ``frame_ms`` frame whose RMS is below ``threshold_dbfs``
    (dB re full scale 1.0); the trailing partial frame is judged on its own.
    Frames at or above the threshold are returned untouched, so speech is
    never altered; only near-silence is made exactly silent."""
    wav = np.asarray(wav, dtype=np.float32)
    n = max(int(round(sr * frame_ms / 1000.0)), 1)
    out = wav.copy()
    thr = 10.0 ** (threshold_dbfs / 20.0)
    for a in range(0, len(wav), n):
        seg = wav[a : a + n]
        if len(seg) and float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) < thr:
            out[a : a + n] = 0.0
    return out


def spans_from_annotation(annotation: Any) -> List[Tuple[float, float]]:
    """``[(start, end)]`` in seconds from a pyannote ``Annotation`` (the VAD
    pipeline output): its timeline's support, i.e. overlapping/adjacent
    speech regions merged, sorted by start."""
    timeline = annotation.get_timeline()
    if hasattr(timeline, "support"):
        timeline = timeline.support()
    return sorted((float(seg.start), float(seg.end)) for seg in timeline)
