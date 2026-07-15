"""``ChannelQualityMetric``: the naturalness/quality leg of the measure-stage
metric battery -- per-IPU MOS prediction, duration-weighted up to a
per-channel then per-window then per-run mean.

Per window (iterating ``meta.scp``), for every GENERATED channel: segment the
channel's ``gen_wav`` into IPUs via ``segments.py``'s dGSLM rule
(``build_ipus``), score EACH IPU with a MOS backend (one backend call per
IPU snippet, not one call on the whole channel -- this mirrors
``ConversationASRMetric``'s per-IPU transcription and keeps the score local
to actual speech rather than diluted by silence). The channel's score is the
SPEECH-DURATION-WEIGHTED mean over its IPU scores (``sum(dur_i * score_i) /
sum(dur_i)``, not a plain mean -- a channel with one long, well-scored IPU
and one short, poorly-scored one should skew toward the long one). The
window's score is the PLAIN (unweighted) mean over its channels' scores
(channels are not re-weighted a second time by their own total speech
duration -- each channel contributes equally to its window regardless of how
much it spoke). The run summary ``utmos_mean`` is the plain mean over
windows. IPUs shorter than ``min_ipu_sec`` (default 0.05 s -- a config knob,
not an empirically validated floor; no real MOS backend/asset was available
locally to calibrate one, same caveat as ``SpeakerDynamicsMetric``'s
``embed_min_sec``) are SKIPPED and counted (``num_ipus_skipped`` in the
per-channel JSONL record), never scored. A channel with zero scoreable IPUs
contributes ``None`` (excluded from the window mean, not a fabricated
score); a window with zero scoreable channels likewise contributes ``None``
to the run summary. Undefined values are always excluded from their mean,
never treated as 0 (same convention as every other metric in this package);
a summary key with zero defined values anywhere is left ``None`` (serializes
as JSON ``null``, rendered as ``-`` by ``local/eval_report.py``) with a
logged warning, never a fabricated 0.0 (see ``_common.py``).

IPU snippets are scored at ``quality_sample_rate`` (default 16000, the same
rate used for VAD/IPU segmentation) rather than at the recipe's native
24 kHz: this repo's own ``evaluate_pseudomos.py`` precedent passes native-
rate audio straight to the UTMOS predictor and lets IT resample internally
to its own native rate (16 kHz, a wav2vec2-based model); pre-resampling to
16 kHz here and telling the predictor ``sr=16000`` reaches the identical
model input without a redundant second resample, so this is a reuse of the
existing VAD-rate audio, not a fidelity loss relative to the repo's other
usage of the same model. Unlike ``SpeakerDynamicsMetric``'s bleed-dB leg,
there is no separate native-rate reload here -- MOS score fidelity at 16 kHz
vs. 24 kHz is a secondary concern for this task's scope.

**A documented deviation from PLAN-step4.md's literal wording, verified
against the real package rather than assumed:** the plan describes the
default backend as "speechmos (utmos22_strong)". The actual PyPI
``speechmos`` package (downloaded and inspected directly: both released
versions, 0.0.1 and 0.0.1.1) contains ONLY ``plcmos``, ``aecmos``, and
``dnsmos`` -- there is no ``utmos22_strong`` module, or any UTMOS module at
all, in that package. The real UTMOS22-strong model lives in a DIFFERENT,
confusingly similarly-named project, ``tarepan/SpeechMOS`` (a GitHub repo,
not a PyPI package), loaded via ``torch.hub.load("tarepan/SpeechMOS:v1.2.0",
"utmos22_strong")`` -- exactly the mechanism this repo's own
``egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_pseudomos.py`` already uses
for the same model. :class:`TorchHubUTMOSBackend` follows that existing
precedent instead of the plan's literal "speechmos" wording. The real,
importable PyPI ``speechmos`` package DOES genuinely provide ``dnsmos.run``,
so the optional DNSMOS backend below (:class:`SpeechMOSDNSMOSBackend`) uses
it as originally intended.

Backends are constructor-injectable; the real defaults are lazy so
constructing this metric (e.g. from ``conf/metrics.yaml`` offline) never
touches the network or loads a model:

* ``mos_backend``: default :class:`TorchHubUTMOSBackend` (UTMOS22-strong via
  ``torch.hub``, see above). ``torch.hub.load`` is called inside
  :meth:`TorchHubUTMOSBackend._load`, invoked from the first
  :meth:`__call__`, never at module scope or in ``__init__``. Unlike the
  other metric classes' soft-imported backends, there is no ``import`` to
  guard here (``torch.hub`` ships with ``torch``, already a hard
  dependency); the network/cache-miss failure mode is instead wrapped with a
  clear, non-silent error pointing at the hub source. This module
  deliberately ships NO live network smoke test for this backend (the
  binding "no GPU/model downloads locally" constraint), unlike
  ``SpeakerDynamicsMetric``'s ``transformers``-gated smoke test -- there is
  no import-based flag to gate on here (``torch`` itself is always
  installed), so exercising the real path would mean an unconditional
  network call in the ordinary test run. Laziness (construction and
  offline-instantiation never call ``torch.hub.load``) is still fully
  covered.
* ``dnsmos_backend``: optional, config-gated (``enable_dnsmos=False`` by
  default; when disabled, no ``dnsmos_backend`` instance is even
  constructed and no ``dnsmos*`` keys appear anywhere). When enabled,
  default :class:`SpeechMOSDNSMOSBackend` wraps the real ``speechmos``
  package's ``dnsmos.run`` (imported inside ``_load``, never at module
  scope), returning the overall MOS (``ovrl_mos``) as a single float per
  whole CHANNEL wav (not per IPU -- kept thin, per the plan's "keep it thin
  ... ship UTMOS-only ... document as future work" escape hatch: DNSMOS is a
  no-reference full-signal predictor, so an IPU-level breakdown would be
  more machinery for a flag the plan already marks optional). If
  ``speechmos`` is missing at real runtime, raises with an install hint --
  no silent fallback.
* ``vad``: default ``segments.VAD()`` (lazy silero), per the shared Task-2
  utility.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import mean_skip_none, summary_value
from .segments import VAD, Interval, build_ipus, load_wav

MOSBackend = Callable[[np.ndarray, int], float]


# --------------------------------------------------------------------------- #
# MOS backends
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
                "ChannelQualityMetric's default MOS backend requires "
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


class SpeechMOSDNSMOSBackend:
    """Optional DNSMOS backend (config-gated, default off): thin wrapper
    around the REAL PyPI ``speechmos`` package's ``dnsmos.run`` (this one
    genuinely exists there, unlike ``utmos22_strong`` -- see module
    docstring). Returns the overall MOS (``ovrl_mos``); expects 16 kHz mono
    audio (the package raises on any other rate). ``speechmos`` is imported
    inside :meth:`_load`, invoked from the first :meth:`__call__`, never at
    module scope or in ``__init__``.
    """

    def __init__(self) -> None:
        self._run = None

    def _load(self) -> None:
        if self._run is not None:
            return
        try:
            from speechmos import dnsmos  # network-free at call time (the
            # package bundles its ONNX weights); network is only needed to
            # `pip install` it, deferred here for laziness consistency with
            # every other backend in this package.
        except ImportError as exc:
            raise ImportError(
                "ChannelQualityMetric's optional DNSMOS backend requires "
                "the `speechmos` package (`pip install speechmos`). Install "
                "it, or inject a `dnsmos_backend=` callable explicitly."
            ) from exc
        self._run = dnsmos.run

    def __call__(self, wav: np.ndarray, sr: int) -> float:
        self._load()
        result = self._run(np.asarray(wav, dtype=np.float32), sr)
        return float(result["ovrl_mos"])


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _weighted_channel_score(ipu_scores: Sequence[Dict[str, float]]) -> Optional[float]:
    """Speech-duration-weighted mean over a channel's per-IPU MOS records
    (each a ``{"duration": ..., "score": ...}`` dict); ``None`` when there
    is nothing to average (no scoreable IPUs, or all durations are zero)."""
    if not ipu_scores:
        return None
    total_dur = sum(r["duration"] for r in ipu_scores)
    if total_dur <= 0:
        return None
    return sum(r["duration"] * r["score"] for r in ipu_scores) / total_dur


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class ChannelQualityMetric(BaseMetric):
    """Naturalness/quality leg of the measure-stage battery: per-IPU MOS,
    duration-weighted per channel, meaned per window and per run. See
    module docstring."""

    def __init__(
        self,
        mos_backend: Optional[MOSBackend] = None,
        vad: Optional[Callable[[np.ndarray, int], Sequence[Interval]]] = None,
        quality_sample_rate: int = 16000,
        min_silence: float = 0.2,
        min_speech: float = 0.0,
        pad: float = 0.0,
        min_ipu_sec: float = 0.05,
        enable_dnsmos: bool = False,
        dnsmos_backend: Optional[MOSBackend] = None,
    ) -> None:
        if dnsmos_backend is not None and not enable_dnsmos:
            raise ValueError(
                "dnsmos_backend was provided but enable_dnsmos=False; the "
                "backend would be silently discarded. Pass enable_dnsmos=True "
                "to use it, or drop dnsmos_backend."
            )
        self.mos_backend = (
            mos_backend if mos_backend is not None else TorchHubUTMOSBackend()
        )
        self.vad = vad if vad is not None else VAD()
        self.quality_sample_rate = quality_sample_rate
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.pad = pad
        self.min_ipu_sec = min_ipu_sec
        self.enable_dnsmos = enable_dnsmos
        self.dnsmos_backend = None
        if self.enable_dnsmos:
            self.dnsmos_backend = (
                dnsmos_backend
                if dnsmos_backend is not None
                else SpeechMOSDNSMOSBackend()
            )

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "channel_quality"
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

    # -- per-channel / per-window scoring ----------------------------------- #
    def _score_channel(self, wav_path: Path) -> Dict[str, Any]:
        wav, sr = load_wav(wav_path, target_sr=self.quality_sample_rate)
        raw_segments = self.vad(wav, sr)
        ipus = build_ipus(
            raw_segments,
            min_silence=self.min_silence,
            min_speech=self.min_speech,
            pad=self.pad,
            total_duration=len(wav) / sr if sr else None,
        )

        ipu_scores: List[Dict[str, float]] = []
        num_skipped = 0
        for start, end in ipus:
            duration = end - start
            if duration < self.min_ipu_sec:
                num_skipped += 1
                continue
            s_samp = max(0, int(round(start * sr)))
            e_samp = min(len(wav), int(round(end * sr)))
            if e_samp <= s_samp:
                num_skipped += 1
                continue
            snippet = wav[s_samp:e_samp]
            score = float(self.mos_backend(snippet, sr))
            ipu_scores.append(
                {"start": start, "end": end, "duration": duration, "score": score}
            )

        record: Dict[str, Any] = {
            "ipu_scores": ipu_scores,
            "num_ipus_skipped": num_skipped,
            "channel_score": _weighted_channel_score(ipu_scores),
        }
        if self.enable_dnsmos:
            record["dnsmos"] = float(self.dnsmos_backend(wav, sr))
        return record

    def _score_window(self, meta: Dict[str, Any], test_dir: Path) -> Dict[str, Any]:
        window_id = meta["window_id"]
        channels_meta = meta["channels"]
        channel_records = [
            self._score_channel(test_dir / ch["gen_wav"]) for ch in channels_meta
        ]
        channel_scores = [c["channel_score"] for c in channel_records]

        record: Dict[str, Any] = {
            "window_id": window_id,
            "num_channels": len(channels_meta),
            "channels": channel_records,
            "utmos_window_mean": mean_skip_none(channel_scores),
        }
        if self.enable_dnsmos:
            dnsmos_vals = [c.get("dnsmos") for c in channel_records]
            record["dnsmos_window_mean"] = mean_skip_none(dnsmos_vals)
        return record

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        summary: Dict[str, Optional[float]] = {
            "utmos_mean": summary_value(
                mean_skip_none(w["utmos_window_mean"] for w in per_window),
                "utmos_mean",
                metric_name=name,
            ),
        }
        if self.enable_dnsmos:
            summary["dnsmos_mean"] = summary_value(
                mean_skip_none(w.get("dnsmos_window_mean") for w in per_window),
                "dnsmos_mean",
                metric_name=name,
            )
        return summary
