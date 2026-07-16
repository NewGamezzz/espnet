"""``QualityMetric``: mix UTMOS, the naturalness/quality leg of the lean
measure-stage metric battery (see PLAN-step4.md's 2026-07-15 revision).

Per window (iterating ``meta.scp``): load the window's ``mix_wav``, resample
to 16 kHz, and score it with ONE MOS backend call -- a single whole-signal
prediction per window, not a per-channel or per-IPU breakdown. The run
summary ``utmos_mean`` is the plain mean over windows.

IPU snippets used to be scored at ``quality_sample_rate`` (default 16000)
rather than the recipe's native 24 kHz; that convention carries over
unchanged to the whole mixdown: this repo's own
``evaluate_pseudomos.py`` precedent passes native-rate audio straight to the
UTMOS predictor and lets IT resample internally to its own native rate
(16 kHz, a wav2vec2-based model), so pre-resampling to 16 kHz here and
telling the predictor ``sr=16000`` reaches the identical model input without
a redundant second resample.

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
precedent instead of the plan's literal "speechmos" wording.

Summary key (one float): ``utmos_mean`` -- mean over windows,
``summary_value``-guarded (a run with zero windows leaves it ``None``,
never a fabricated 0.0).

Deferred to a later PR (see README.md's "Deferred to the next PR" list):
per-IPU UTMOS weighting (this metric used to VAD-segment each channel and
duration-weight per-IPU scores; the reworked infer stage's whole-window
generation plus the mixdown-only scope makes that unnecessary complexity)
and DNSMOS -- both cut in the 2026-07-15 PR #10 review to keep this battery
lean and easy to review.

Backend is constructor-injectable; the real default is lazy so constructing
this metric (e.g. from ``conf/metrics.yaml`` offline) never touches the
network or loads a model:

* ``mos_backend``: default :class:`TorchHubUTMOSBackend` (UTMOS22-strong via
  ``torch.hub``, see above). ``torch.hub.load`` is called inside
  :meth:`TorchHubUTMOSBackend._load`, invoked from the first
  :meth:`__call__`, never at module scope or in ``__init__``. Unlike the
  other metric classes' soft-imported backends, there is no ``import`` to
  guard here (``torch.hub`` ships with ``torch``, already a hard
  dependency); the network/cache-miss failure mode is instead wrapped with a
  clear, non-silent error pointing at the hub source. This module
  deliberately ships NO live network smoke test for this backend (the
  binding "no GPU/model downloads locally" constraint) -- laziness
  (construction and offline-instantiation never call ``torch.hub.load``) is
  still fully covered.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import load_wav, mean_skip_none, summary_value

MOSBackend = Callable[[np.ndarray, int], float]


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
# metric
# --------------------------------------------------------------------------- #
class QualityMetric(BaseMetric):
    """Naturalness/quality leg of the lean measure-stage battery: one UTMOS
    call per window on the mixdown. See module docstring."""

    def __init__(
        self,
        mos_backend: Optional[MOSBackend] = None,
        quality_sample_rate: int = 16000,
    ) -> None:
        self.mos_backend = (
            mos_backend if mos_backend is not None else TorchHubUTMOSBackend()
        )
        self.quality_sample_rate = quality_sample_rate

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
        wav, sr = load_wav(
            test_dir / meta["mix_wav"], target_sr=self.quality_sample_rate
        )
        score = float(self.mos_backend(wav, sr))
        return {"window_id": meta["window_id"], "score": score}

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        return {
            "utmos_mean": summary_value(
                mean_skip_none(w["score"] for w in per_window),
                "utmos_mean",
                metric_name=name,
            ),
        }
