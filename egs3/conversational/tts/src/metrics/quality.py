"""``QualityMetric``: per-IPU UTMOS (primary) plus mix UTMOS, the
naturalness/quality leg of the measure-stage metric battery.

An IPU (interpausal unit) is a stretch of one speaker's speech bounded by
silences of at least 200 ms -- the dGSLM convention (arXiv 2203.16502).
Silero VAD segments each channel's generated wav into IPUs, and every IPU of
at least ``min_ipu_sec`` seconds is scored with one MOS-backend call.

Why VAD units instead of the manifest turn spans (first-Delta-run finding,
2026-07-16): manifest spans describe the GROUND-TRUTH alignment, and only
``gt``/``resynth`` audio follows it. Generated audio places speech at times
of the model's own choosing, so ground-truth spans would slice generated
words mid-syllable and score silence as speech. VAD derives the unit from
the audio actually being scored, which keeps one identical definition across
all conditions (anchors are only anchors if they flow through the same
pipeline). Validated on Delta: per-IPU vs manifest-per-turn UTMOS agreed
within ~0.1 on both anchors.

Why per-IPU at all: UTMOS is sensitive to the scoring unit. Whole windows
and whole channels embed long silences and (in the mix) overlapping
speakers, which UTMOS punishes regardless of audio quality; isolated speech
segments are also the definition under which the SSSD corpus paper reports
UTMOS (2.55 +/- 0.72 per utterance, Table 3 of Sheikh et al., Interspeech
2025), so per-IPU scores are comparable to the corpus baseline.

Audio is pre-resampled to ``quality_sample_rate`` (default 16000) and the
predictor is told ``sr=16000``: this repo's ``evaluate_pseudomos.py``
precedent lets the wav2vec2-based UTMOS model resample internally to its
native 16 kHz, so pre-resampling here reaches the identical model input
without a redundant second resample. 16 kHz is also the rate Silero VAD
operates on, so both backends share one loaded wav per channel.

Summary keys (``summary_value``-guarded: undefined stays ``None``, never a
fabricated 0.0):

* ``utmos_ipu_mean`` -- mean over every scored IPU of every channel of every
  window (pooled over the run, not a mean of window means). PRIMARY quality
  number; read it against the gt/resynth anchors, never as absolute quality
  (UTMOS rewards clean read-style speech and punishes spontaneity -- on the
  first Delta run the zero-gate pretrained model out-scored ground truth).
* ``utmos_mix_mean`` -- one whole-mixdown score per window, plain mean over
  windows. Kept from the lean v1 battery for cross-run continuity.
* ``ipu_count`` -- number of scored IPUs in the run. Diagnostic: on the
  first Delta run the pretrained model produced ~2x ground truth's count
  (1241 vs 653) by filling the forced window duration with speech.

**A documented deviation from PLAN-step4.md's literal wording, verified
against the real package rather than assumed:** the plan describes the
default MOS backend as "speechmos (utmos22_strong)". The actual PyPI
``speechmos`` package (both released versions, 0.0.1 and 0.0.1.1) contains
ONLY ``plcmos``, ``aecmos``, and ``dnsmos``. The real UTMOS22-strong model
lives in the ``tarepan/SpeechMOS`` GitHub repo, loaded via
``torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`` -- the
mechanism this repo's ``evaluate_pseudomos.py`` already uses.

Backends are constructor-injectable and the real defaults are lazy, so
constructing this metric (e.g. from ``conf/metrics.yaml`` offline) never
touches the network or loads a model:

* ``mos_backend``: default :class:`TorchHubUTMOSBackend`;
  ``torch.hub.load`` happens inside :meth:`TorchHubUTMOSBackend._load` on
  the first call, never at module scope or in ``__init__``.
* ``vad_backend``: default :class:`SileroVADSegmenter`, backed by
  faster-whisper's bundled Silero VAD (ONNX; already a hard dependency of
  ``ConversationASRMetric``'s default transcriber, so no new package).
  ``faster_whisper`` is imported inside :meth:`SileroVADSegmenter._load`,
  invoked from the first call.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import load_wav, mean_skip_none, summary_value

MOSBackend = Callable[[np.ndarray, int], float]
# (wav, sr) -> [(start_sec, end_sec), ...] speech spans in seconds.
VADBackend = Callable[[np.ndarray, int], Sequence[Tuple[float, float]]]


# --------------------------------------------------------------------------- #
# MOS backend
# --------------------------------------------------------------------------- #
class TorchHubUTMOSBackend:
    """Real default MOS backend: UTMOS22-strong via
    ``torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`` -- see
    module docstring for why this is the correct source, not the PyPI
    ``speechmos`` package. Deferred to the first :meth:`__call__`;
    constructing this class is always safe offline.
    """

    def __init__(
        self,
        repo: str = "tarepan/SpeechMOS:v1.2.0",
        model_name: str = "utmos22_strong",
        device: str = "cpu",
    ) -> None:
        self.repo = repo
        self.model_name = model_name
        self.device = device
        self._predictor = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            predictor = torch.hub.load(self.repo, self.model_name)
        except Exception as exc:  # pragma: no cover - network/cache dependent
            raise RuntimeError(
                "QualityMetric's default MOS backend requires "
                f"torch.hub.load({self.repo!r}, {self.model_name!r}) to "
                "succeed (network access on first call, or a pre-warmed "
                "torch hub cache, e.g. on Delta). Warm the cache, or inject "
                "a `mos_backend=` callable explicitly -- silently falling "
                "back to a weaker predictor would corrupt cross-run UTMOS "
                "comparability."
            ) from exc
        self._predictor = predictor.to(self.device)

    def __call__(self, wav: np.ndarray, sr: int) -> float:
        self._load()
        wav_t = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            score = self._predictor(wav_t.unsqueeze(0), sr)
        return float(score.reshape(-1)[0].item())


# --------------------------------------------------------------------------- #
# VAD backend
# --------------------------------------------------------------------------- #
class SileroVADSegmenter:
    """Real default VAD backend: Silero VAD as bundled with faster-whisper
    (ONNX runtime; no torch.hub fetch, and faster-whisper is already the
    ASR metric's default backend). Returns speech spans in seconds.

    ``min_silence_duration_ms=200`` is the dGSLM IPU rule: speech separated
    by less than 200 ms of silence merges into one IPU.

    ``faster_whisper`` is imported inside :meth:`_load`, invoked from the
    first :meth:`__call__`, never at module scope or in ``__init__`` --
    constructing this class (e.g. as the metric's default) is always safe
    offline.
    """

    #: The only sample rate Silero VAD (and this segmenter) operates on.
    SAMPLE_RATE = 16000

    def __init__(
        self,
        min_silence_duration_ms: int = 200,
        min_speech_duration_ms: int = 250,
    ) -> None:
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self._get_speech_timestamps = None
        self._options = None

    def _load(self) -> None:
        if self._get_speech_timestamps is not None:
            return
        try:
            from faster_whisper.vad import (  # network-free, ships in the wheel
                VadOptions,
                get_speech_timestamps,
            )
        except ImportError as exc:
            raise ImportError(
                "QualityMetric's default VAD backend requires faster-whisper "
                "(`pip install faster-whisper`) for its bundled Silero VAD. "
                "Install it, or inject a `vad_backend=` callable explicitly."
            ) from exc
        self._get_speech_timestamps = get_speech_timestamps
        self._options = VadOptions(
            min_silence_duration_ms=self.min_silence_duration_ms,
            min_speech_duration_ms=self.min_speech_duration_ms,
        )

    def __call__(self, wav: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        if sr != self.SAMPLE_RATE:
            raise ValueError(
                f"SileroVADSegmenter operates on {self.SAMPLE_RATE} Hz audio, "
                f"got {sr} -- resample before segmenting"
            )
        self._load()
        spans = self._get_speech_timestamps(
            np.asarray(wav, dtype=np.float32), vad_options=self._options
        )
        return [
            (span["start"] / self.SAMPLE_RATE, span["end"] / self.SAMPLE_RATE)
            for span in spans
        ]


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class QualityMetric(BaseMetric):
    """Quality leg of the measure-stage battery: UTMOS per VAD-derived IPU
    on every channel (primary) plus one whole-mixdown UTMOS per window.
    See module docstring."""

    def __init__(
        self,
        mos_backend: Optional[MOSBackend] = None,
        vad_backend: Optional[VADBackend] = None,
        quality_sample_rate: int = 16000,
        min_ipu_sec: float = 1.0,
    ) -> None:
        self.mos_backend = (
            mos_backend if mos_backend is not None else TorchHubUTMOSBackend()
        )
        self.vad_backend = (
            vad_backend if vad_backend is not None else SileroVADSegmenter()
        )
        self.quality_sample_rate = quality_sample_rate
        self.min_ipu_sec = min_ipu_sec

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "quality"
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
    def _score_window(self, meta: Dict[str, Any], test_dir: Path) -> Dict[str, Any]:
        sr = self.quality_sample_rate

        mix_wav, mix_sr = load_wav(test_dir / meta["mix_wav"], target_sr=sr)
        mix_score = float(self.mos_backend(mix_wav, mix_sr))

        ipus: List[Dict[str, Any]] = []
        n_skipped_short = 0
        for ch_index, ch in enumerate(meta["channels"]):
            wav, ch_sr = load_wav(test_dir / ch["gen_wav"], target_sr=sr)
            for start, end in self.vad_backend(wav, ch_sr):
                if end - start < self.min_ipu_sec:
                    n_skipped_short += 1
                    continue
                segment = wav[int(start * ch_sr) : int(end * ch_sr)]
                ipus.append(
                    {
                        "channel": ch_index,
                        "start": round(float(start), 3),
                        "end": round(float(end), 3),
                        "score": float(self.mos_backend(segment, ch_sr)),
                    }
                )

        return {
            "window_id": meta["window_id"],
            "mix_score": mix_score,
            "ipus": ipus,
            "n_ipus_skipped_short": n_skipped_short,
        }

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        all_ipus = [ipu for w in per_window for ipu in w["ipus"]]
        return {
            "utmos_ipu_mean": summary_value(
                mean_skip_none(ipu["score"] for ipu in all_ipus),
                "utmos_ipu_mean",
                metric_name=name,
            ),
            "utmos_mix_mean": summary_value(
                mean_skip_none(w["mix_score"] for w in per_window),
                "utmos_mix_mean",
                metric_name=name,
            ),
            "ipu_count": summary_value(
                float(len(all_ipus)) if per_window else None,
                "ipu_count",
                metric_name=name,
            ),
        }
