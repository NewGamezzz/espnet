"""``SpeakerSimilarityMetric``: SIM-o (type 1), the speaker-identity leg of
the lean measure-stage metric battery (see PLAN-step4.md's 2026-07-15
revision).

Per window (iterating ``meta.scp``), for every channel ``k``: embed
``channels[k].prompt_wav`` (the channel's acoustic prompt, ONE solo turn,
whole file) and ``channels[k].gen_wav`` (the ENTIRE generated channel, whole
file -- the reworked infer stage now generates the whole window, not just a
region after a boundary) with the injectable embedder, and take their cosine
similarity. No VAD, no segmentation, no per-IPU anything: both sides are
embedded as a single whole-file snippet, mirroring typical SV-embedding
practice (embed one enrollment/test utterance rather than averaging many
short ones).

``embed_min_sec`` floor: a wav (either side) shorter than ``embed_min_sec``
(default 0.3 s -- a config knob, not an empirically validated lower bound;
WavLM-SV-family x-vector embeddings are unreliable on sub-word snippets) is
NOT embedded; that channel's cosine is ``None`` for that window
(skip-and-count, ``n_skipped`` in the per-window JSONL detail -- never a
fabricated 0.0, and never counted toward ``sim_o_mean``).

Summary key (one float): ``sim_o_mean`` -- the mean over EVERY (window,
channel) cosine in the run (a flat mean, not a mean-of-per-window-means),
``summary_value``-guarded so a run where every channel was skipped leaves it
``None``.

Deferred to a later PR (see README.md's "Deferred to the next PR" list):
cross-turn consistency, drift, cross-channel confusion, and generated bleed
dB -- all cut in the 2026-07-15 PR #10 review to keep this battery lean and
easy to review (they depended on the now-deleted VAD/IPU machinery and
ground-truth solo-span bookkeeping).

Backend is constructor-injectable; the real default is lazy so constructing
this metric (e.g. from ``conf/metrics.yaml`` offline) never touches the
network or loads a model:

* ``embedder``: default :class:`WavLMSVEmbedder`, a WavLM-based
  speaker-verification x-vector via ``transformers``' ``WavLMForXVector``.
  The checkpoint TAG is a config knob (``model_tag``, default
  ``"microsoft/wavlm-base-plus-sv"`` -- the accessible/small member of the
  WavLM-SV family, not necessarily the checkpoint used for real runs).
  ``transformers`` is imported inside :meth:`WavLMSVEmbedder._load`, invoked
  from the first ``__call__``, never at module scope or in ``__init__``.

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

# --------------------------------------------------------------------------- #
# embedder backend
# --------------------------------------------------------------------------- #
Embedder = Callable[[np.ndarray, int], np.ndarray]


class WavLMSVEmbedder:
    """Real default embedder: a WavLM-based speaker-verification x-vector via
    ``transformers``' ``WavLMForXVector``.

    ``transformers`` (and the checkpoint download) is imported inside
    :meth:`_load`, invoked from the first :meth:`__call__`, never at module
    scope or in ``__init__`` -- constructing this class (e.g. as a metric's
    default) is always safe offline. Expects 16 kHz mono float audio (the
    rate WavLM-SV checkpoints are pretrained at); a mismatched rate raises
    before touching the model.
    """

    def __init__(
        self, model_tag: str = "microsoft/wavlm-base-plus-sv", device: str = "cpu"
    ) -> None:
        self.model_tag = model_tag
        self.device = device
        self._model = None
        self._feature_extractor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import (  # network/asset fetch, deferred
            AutoFeatureExtractor,
            WavLMForXVector,
        )

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_tag)
        self._model = WavLMForXVector.from_pretrained(self.model_tag).to(self.device)
        self._model.eval()

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            raise ValueError(f"WavLMSVEmbedder expects 16000 Hz audio, got {sr}")
        self._load()
        inputs = self._feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            output = self._model(input_values)
        embedding = torch.nn.functional.normalize(output.embeddings[0], dim=-1)
        return embedding.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# pure math helper
# --------------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class SpeakerSimilarityMetric(BaseMetric):
    """Speaker-identity leg of the lean measure-stage battery: SIM-o (type
    1), prompt vs whole generated channel. See module docstring."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        embed_sample_rate: int = 16000,
        embed_min_sec: float = 0.3,
        embed_max_sec: Optional[float] = None,
        embed_silence_rms: float = 1e-3,
    ) -> None:
        self.embedder = embedder if embedder is not None else WavLMSVEmbedder()
        self.embed_sample_rate = embed_sample_rate
        self.embed_min_sec = embed_min_sec
        # Long-form channels (an AMI meeting is ~35 min) cannot go through
        # WavLM in one pass (19.6 GB conv1d, measured).  With ``embed_max_sec``
        # set, audio longer than that is embedded in consecutive segments of
        # at most that length, near-silent segments (RMS below
        # ``embed_silence_rms``, i.e. masked ground truth or an idle channel)
        # are skipped, and the L2-normalized segment embeddings are averaged
        # and re-normalized.  None keeps the single-pass path bit-identical.
        self.embed_max_sec = embed_max_sec
        self.embed_silence_rms = embed_silence_rms

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "speaker_similarity"
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
        window_id = meta["window_id"]
        channels_meta = meta["channels"]

        channel_records = []
        n_skipped = 0
        for ch in channels_meta:
            gen_wav, gsr = load_wav(
                test_dir / ch["gen_wav"], target_sr=self.embed_sample_rate
            )
            prompt_wav, psr = load_wav(
                test_dir / ch["prompt_wav"], target_sr=self.embed_sample_rate
            )
            gen_dur = len(gen_wav) / gsr if gsr else 0.0
            prompt_dur = len(prompt_wav) / psr if psr else 0.0

            if gen_dur < self.embed_min_sec or prompt_dur < self.embed_min_sec:
                channel_records.append({"cosine": None, "skipped": True})
                n_skipped += 1
                continue

            gen_embedding = self._embed(gen_wav, gsr)
            prompt_embedding = self._embed(prompt_wav, psr)
            cosine = _cosine(gen_embedding, prompt_embedding)
            channel_records.append({"cosine": cosine, "skipped": False})

        return {
            "window_id": window_id,
            "num_channels": len(channels_meta),
            "channels": channel_records,
            "n_skipped": n_skipped,
        }

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        all_cosines = [c["cosine"] for w in per_window for c in w["channels"]]
        return {
            "sim_o_mean": summary_value(
                mean_skip_none(all_cosines), "sim_o_mean", metric_name=name
            ),
        }

    # -- embedding ----------------------------------------------------------- #
    def _embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav)
        if self.embed_max_sec is None or wav.shape[0] <= int(self.embed_max_sec * sr):
            return np.asarray(self.embedder(wav, sr), dtype=np.float64)
        seg = int(self.embed_max_sec * sr)
        pieces = [wav[i : i + seg] for i in range(0, wav.shape[0], seg)]
        min_len = int(self.embed_min_sec * sr)
        embeddings = []
        for piece in pieces:
            if piece.shape[0] < min_len:
                continue
            rms = float(np.sqrt(np.mean(np.square(piece.astype(np.float64)))))
            if rms < self.embed_silence_rms:
                continue
            e = np.asarray(self.embedder(piece, sr), dtype=np.float64)
            norm = float(np.linalg.norm(e))
            embeddings.append(e / norm if norm > 1e-12 else e)
        if not embeddings:
            # Nothing audible anywhere: embed the loudest piece so the metric
            # still returns a vector (cosine will be near-meaningless, as it
            # should be for a silent channel).
            loudest = max(pieces, key=lambda x: float(np.mean(np.square(x.astype(np.float64)))))
            return np.asarray(self.embedder(loudest, sr), dtype=np.float64)
        mean = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-12 else mean
