"""``SpeakerDynamicsMetric``: the speaker-identity leg of the measure-stage
metric battery.

Per window (iterating ``meta.scp``), for every channel ``k``:

1. **SIM-o** -- cosine similarity between the embedding of channel ``k``'s
   GENERATED speech and the embedding of channel ``k``'s acoustic PROMPT wav.
   "Channel ``k``'s generated speech" is defined as the CONCATENATION of that
   channel's generated-region VAD IPUs (dGSLM rule, ``segments.py``), embedded
   as a single snippet -- not a per-IPU mean of embeddings. This mirrors
   typical SV-embedding practice (embed one longer enrollment/test utterance
   rather than averaging many short ones) and gives the concatenation more
   usable duration than any individual IPU. The prompt side is embedded as
   the WHOLE prompt wav with NO VAD applied (the acoustic prompt is ground
   truth audio for that channel; embedding models are far less sensitive to
   incidental internal silence than an ASR transcriber is).
2. **Cross-turn consistency / drift** -- EACH generated IPU is ALSO embedded
   individually (a second, separate set of embedder calls from item 1's
   concatenation). ``sim_consistency`` is the mean pairwise cosine among a
   channel's per-IPU embeddings (channels with fewer than 2 embedded IPUs
   contribute no value, not a fabricated 0). The drift curve is, per IPU (in
   time order), the cosine of that IPU's own embedding to the channel's
   PROMPT embedding; ``sim_drift_slope`` is the least-squares slope of that
   curve against IPU index (channels with fewer than 2 DEFINED points
   contribute no slope). The full curve (one entry per IPU, ``None`` for a
   skipped IPU -- see the ``embed_min_sec`` floor below) is written to the
   per-window JSONL for inspection; only the scalar slope is summarized.
3. **Cross-channel confusion** -- cosine(embedding of gen channel ``i``'s
   concatenated speech [same embedding as item 1], embedding of channel
   ``j``'s prompt) for every ordered pair ``i != j``; ``confusion_mean`` is
   the mean over pairs that had BOTH embeddings defined.
4. **Generated bleed dB** -- reuses ``segments.py``'s re-exported
   ``solo_regions``/``merge_intervals`` (from ``local/crosstalk_report.py``)
   on the meta JSON's GROUND-TRUTH turn spans (``meta["turns"]``), not VAD on
   the generated audio: a channel ``j`` is "solo" wherever the SCRIPT says
   only ``j`` is talking. ``meta["turns"]`` carries WINDOW-relative time (the
   same clock as ``prompt_boundary_sec``; see ``src/inference.py``'s
   ``_turn_spans`` docstring), while every ``gen_wav`` covers only the
   GENERATED region, so turns are shifted by ``-prompt_boundary_sec`` and
   clipped to ``[0, window_duration_sec - prompt_boundary_sec]`` before
   ``solo_regions`` sees them (:func:`_clip_turns_to_region`). ``meta`` does
   not carry the model's hop size, so this uses ``prompt_boundary_sec``
   directly rather than the frame-exact cut ``src/inference.py`` documents
   (``prompt_boundary_frames * hop``); the resulting sub-hop error is well
   under the ``bleed_guard`` edge trim (default 0.2 s) and therefore never
   changes which samples are measured. For every ordered pair ``(k, j)``,
   ``k != j``, this measures channel ``k``'s energy
   in the GENERATED audio during channel ``j``'s solo spans, relative to
   channel ``j``'s own energy there, mirroring ``crosstalk_report.py``'s
   ``bleed_db(k <- j) = 10*log10(E[ch_k^2 | solo j] / E[ch_j^2 | solo j])`` --
   except here both sides come from the model's OUTPUT, so a nonzero value
   means the model put energy on a channel that the script says should have
   been silent, not a microphone-bleed artifact. A pair with no solo region
   for ``j`` (or one entirely swallowed by the edge ``guard``) is SKIPPED,
   not scored as 0 dB, and recorded separately so the omission is visible.
   Energy sums use the NATIVE-rate ``gen_wav`` (a fresh, unresampled load),
   not the 16 kHz copy used for VAD/embedding, to avoid a resampling filter
   perturbing the energy ratio. ``bleed_db_p50``/``bleed_db_p90`` are
   percentiles (``numpy.percentile``, linear interpolation) over the FLAT
   list of every scored pair's ``bleed_db`` across ALL windows -- unlike the
   other summary keys, this is NOT a mean-of-per-window-means, because the
   population of interest is "how bad is a bleed pair", not "how bad is a
   window".

Summary keys (six floats):

* ``sim_o_mean``       -- per-window mean of per-channel SIM-o, meaned over
  windows.
* ``sim_consistency``  -- per-window mean of per-channel consistency, meaned
  over windows.
* ``sim_drift_slope``  -- per-window mean of per-channel drift slope, meaned
  over windows.
* ``confusion_mean``   -- per-window mean over confusion pairs, meaned over
  windows.
* ``bleed_db_p50``, ``bleed_db_p90`` -- percentiles over ALL scored bleed
  pairs, pooled across every window (see item 4 above).

Windows/channels/pairs with an undefined ("None") value are excluded from
every mean, not treated as 0 (same convention as ``ConversationASRMetric``).
If a summary key has NO defined value across the whole run, it falls back to
0.0 with a logged warning.

``embed_min_sec`` floor and its documented asymmetry: an IPU (or the whole
prompt) shorter than ``embed_min_sec`` (default 0.3 s) is NOT individually
embedded -- WavLM-SV-family x-vector embeddings are unreliable on sub-word
snippets, and this threshold is a config knob, not an empirically validated
lower bound (no real embedder/asset is available in this environment to
calibrate one). The floor applies independently to: (a) each IPU considered
for PER-IPU embedding (consistency/drift) -- a too-short IPU contributes
``None`` at its own index and is excluded from both, but STILL contributes
its raw audio to the channel's CONCATENATED embedding input; (b) the total
CONCATENATED duration for SIM-o/confusion -- if that whole concatenation is
still under the floor (e.g. the channel has no IPUs at all, or only a
handful of very short ones), the channel's ``gen_embedding`` is ``None`` and
SIM-o/confusion involving it are skipped; (c) the whole prompt wav -- an
unusually short prompt yields ``prompt_embedding = None``.

Backends are constructor-injectable; the real default is lazy so
constructing this metric (e.g. from ``conf/metrics.yaml`` offline) never
touches the network or loads a model:

* ``embedder``: default :class:`WavLMSVEmbedder`, a WavLM-based
  speaker-verification x-vector via ``transformers``' ``WavLMForXVector``.
  The checkpoint TAG is a config knob (``model_tag``, default
  ``"microsoft/wavlm-base-plus-sv"`` -- the accessible/small member of the
  WavLM-SV family, not necessarily the checkpoint used for real runs); the
  exact tag for the first real Delta run is pinned there, not here.
  ``transformers`` is imported inside :meth:`WavLMSVEmbedder._load`, invoked
  from the first ``__call__``, never at module scope or in ``__init__``.
* ``vad``: default ``segments.VAD()`` (lazy silero), per the shared Task-2
  utility, used ONLY for the generated-audio IPU segmentation (items 1-3
  above); the bleed-dB leg (item 4) never calls VAD.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from espnet3.components.metrics.base_metric import BaseMetric

from .segments import Interval, VAD, build_ipus, load_wav, merge_intervals, solo_regions

logger = logging.getLogger(__name__)

_POWER_EPS = 1e-12


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
    before touching the model, same style as ``segments.SileroVADBackend``.
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
# small pure-function helpers
# --------------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _mean_skip_none(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fallback_zero(value: Optional[float], key: str) -> float:
    if value is None:
        logger.warning(
            "SpeakerDynamicsMetric: no window produced a defined value for "
            "'%s'; defaulting the run summary to 0.0",
            key,
        )
        return 0.0
    return float(value)


def _sec_to_sample(t: float, sr: int) -> int:
    return int(round(t * sr))


# --------------------------------------------------------------------------- #
# per-channel features (VAD/IPU + embeddings), and the pure aggregations
# over them (items 1-3 of the module docstring)
# --------------------------------------------------------------------------- #
@dataclass
class ChannelFeatures:
    """One channel's inputs to items 1-3. ``per_ipu_embeddings`` is aligned
    1:1 with ``ipus`` (same length, same order); an entry is ``None`` when
    that IPU was below ``embed_min_sec`` and therefore never individually
    embedded (see module docstring)."""

    ipus: List[Interval] = field(default_factory=list)
    gen_embedding: Optional[np.ndarray] = None
    prompt_embedding: Optional[np.ndarray] = None
    per_ipu_embeddings: List[Optional[np.ndarray]] = field(default_factory=list)


def _sim_o(features: ChannelFeatures) -> Optional[float]:
    if features.gen_embedding is None or features.prompt_embedding is None:
        return None
    return _cosine(features.gen_embedding, features.prompt_embedding)


def _consistency(features: ChannelFeatures) -> Optional[float]:
    embeddings = [e for e in features.per_ipu_embeddings if e is not None]
    if len(embeddings) < 2:
        return None
    sims = [_cosine(a, b) for a, b in itertools.combinations(embeddings, 2)]
    return _mean(sims)


def _drift(features: ChannelFeatures) -> Tuple[List[Optional[float]], Optional[float]]:
    """Returns ``(curve, slope)``. ``curve[i]`` is the cosine of IPU ``i``'s
    own embedding to the prompt embedding (``None`` where the IPU has no
    embedding, or the prompt embedding itself is undefined). ``slope`` is the
    least-squares fit of the DEFINED curve points against their ORIGINAL IPU
    index (i.e. skipped IPUs leave a hole in ``x``, they are not
    renumbered-away), or ``None`` with fewer than 2 defined points."""
    if features.prompt_embedding is None:
        return [None] * len(features.per_ipu_embeddings), None

    curve = [
        _cosine(e, features.prompt_embedding) if e is not None else None
        for e in features.per_ipu_embeddings
    ]
    defined = [(i, c) for i, c in enumerate(curve) if c is not None]
    if len(defined) < 2:
        return curve, None
    xs = [i for i, _ in defined]
    ys = [c for _, c in defined]
    return curve, _least_squares_slope(xs, ys)


def _confusion_pairs(features: Sequence[ChannelFeatures]) -> List[Dict[str, Any]]:
    """Ordered-pair cross-channel confusion: gen channel ``i`` vs prompt
    channel ``j``, ``i != j``. Pairs missing either embedding are omitted."""
    pairs: List[Dict[str, Any]] = []
    n = len(features)
    for i in range(n):
        gi = features[i].gen_embedding
        if gi is None:
            continue
        for j in range(n):
            if i == j:
                continue
            pj = features[j].prompt_embedding
            if pj is None:
                continue
            pairs.append(
                {"gen_channel": i, "prompt_channel": j, "cosine": _cosine(gi, pj)}
            )
    return pairs


# --------------------------------------------------------------------------- #
# generated-bleed dB (item 4 of the module docstring)
# --------------------------------------------------------------------------- #
def _clip_turns_to_region(
    turns: Sequence[Dict[str, Any]], boundary_sec: float, region_duration: float
) -> Dict[int, List[Interval]]:
    """GT turn spans (window-relative, per ``src/inference.py``'s
    ``_turn_spans``) -> per-channel intervals relative to the GENERATED
    region's own start (``gen_wav``'s timeline), clipped to
    ``[0, region_duration]``. A turn entirely outside the generated region
    (fully in the prompt, or past the window end) contributes nothing; a
    turn straddling the boundary is clipped to its in-region portion."""
    per_channel: Dict[int, List[Interval]] = {}
    for turn in turns:
        channel = int(turn["channel"])
        start = float(turn["start"]) - boundary_sec
        end = float(turn["end"]) - boundary_sec
        clipped_start = max(0.0, start)
        clipped_end = min(region_duration, end)
        if clipped_end <= clipped_start:
            continue
        per_channel.setdefault(channel, []).append((clipped_start, clipped_end))
    return per_channel


def _region_sumsq(
    wav: np.ndarray, sr: int, regions: Sequence[Interval]
) -> Tuple[float, int]:
    """Sum of squares and sample count of ``wav`` over ``regions`` (seconds),
    clamped to ``wav``'s own length."""
    total = 0.0
    frames = 0
    for start, end in regions:
        s_samp = max(0, _sec_to_sample(start, sr))
        e_samp = min(len(wav), _sec_to_sample(end, sr))
        if e_samp <= s_samp:
            continue
        block = wav[s_samp:e_samp].astype(np.float64)
        total += float(np.sum(block**2))
        frames += block.shape[0]
    return total, frames


def _bleed_pairs(
    native_wavs: Sequence[Tuple[np.ndarray, int]],
    turns: Sequence[Dict[str, Any]],
    boundary_sec: float,
    region_duration: float,
    guard: float,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """Returns ``(pairs, skipped)``; ``pairs`` entries carry
    ``gen_channel`` (k), ``solo_channel`` (j), ``bleed_db``, ``solo_sec``.
    ``skipped`` lists ``(k, j)`` pairs with no usable solo region for ``j``."""
    n = len(native_wavs)
    raw = _clip_turns_to_region(turns, boundary_sec, region_duration)
    per_channel_intervals = {c: merge_intervals(raw.get(c, [])) for c in range(n)}

    pairs: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, int]] = []
    for j in range(n):
        regions = solo_regions(per_channel_intervals, j, guard)
        if not regions:
            skipped.extend((k, j) for k in range(n) if k != j)
            continue
        solo_sec = sum(e - s for s, e in regions)
        wav_j, sr_j = native_wavs[j]
        sumsq_j, frames_j = _region_sumsq(wav_j, sr_j, regions)
        for k in range(n):
            if k == j:
                continue
            wav_k, sr_k = native_wavs[k]
            sumsq_k, frames_k = _region_sumsq(wav_k, sr_k, regions)
            if frames_j == 0 or frames_k == 0:
                skipped.append((k, j))
                continue
            power_j = sumsq_j / frames_j
            power_k = sumsq_k / frames_k
            bleed_db = 10.0 * math.log10(
                max(power_k, _POWER_EPS) / max(power_j, _POWER_EPS)
            )
            pairs.append(
                {
                    "gen_channel": k,
                    "solo_channel": j,
                    "bleed_db": bleed_db,
                    "solo_sec": solo_sec,
                }
            )
    return pairs, skipped


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class SpeakerDynamicsMetric(BaseMetric):
    """Speaker-identity leg of the measure-stage battery: SIM-o, cross-turn
    consistency/drift, cross-channel confusion, generated bleed dB."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vad: Optional[Callable[[np.ndarray, int], Sequence[Interval]]] = None,
        embed_sample_rate: int = 16000,
        min_silence: float = 0.2,
        min_speech: float = 0.0,
        pad: float = 0.0,
        embed_min_sec: float = 0.3,
        bleed_guard: float = 0.2,
    ) -> None:
        self.embedder = embedder if embedder is not None else WavLMSVEmbedder()
        self.vad = vad if vad is not None else VAD()
        self.embed_sample_rate = embed_sample_rate
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.pad = pad
        self.embed_min_sec = embed_min_sec
        self.bleed_guard = bleed_guard

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "speaker_dynamics"
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
        boundary_sec = float(meta["prompt_boundary_sec"])
        channels_meta = meta["channels"]
        turns = meta.get("turns", [])
        n = len(channels_meta)
        region_duration = float(meta["window_duration_sec"]) - boundary_sec

        features: List[ChannelFeatures] = []
        native_wavs: List[Tuple[np.ndarray, int]] = []
        for ch in channels_meta:
            gen_path = test_dir / ch["gen_wav"]
            features.append(
                self._channel_features(gen_path, test_dir / ch["prompt_wav"])
            )
            # Fresh, unresampled load for bleed-dB energy sums (see module
            # docstring: resampling must not perturb the energy ratio).
            native_wavs.append(load_wav(gen_path))

        sim_o = [_sim_o(f) for f in features]
        consistency = [_consistency(f) for f in features]
        drift_curves: List[List[Optional[float]]] = []
        drift_slopes: List[Optional[float]] = []
        for f in features:
            curve, slope = _drift(f)
            drift_curves.append(curve)
            drift_slopes.append(slope)
        confusion_pairs = _confusion_pairs(features)
        bleed_pairs, bleed_skipped = _bleed_pairs(
            native_wavs, turns, boundary_sec, region_duration, self.bleed_guard
        )

        return {
            "window_id": window_id,
            "num_channels": n,
            "sim_o": sim_o,
            "sim_o_mean": _mean_skip_none(sim_o),
            "consistency": consistency,
            "sim_consistency": _mean_skip_none(consistency),
            "drift_curve": drift_curves,
            "drift_slope": drift_slopes,
            "sim_drift_slope": _mean_skip_none(drift_slopes),
            "confusion_pairs": confusion_pairs,
            "confusion_mean": _mean([p["cosine"] for p in confusion_pairs]),
            "bleed_pairs": bleed_pairs,
            "bleed_skipped_pairs": [list(p) for p in bleed_skipped],
        }

    def _summarize(self, per_window: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        def agg(key: str) -> Optional[float]:
            return _mean_skip_none(w[key] for w in per_window)

        all_bleed_db = [
            pair["bleed_db"] for w in per_window for pair in w["bleed_pairs"]
        ]
        return {
            "sim_o_mean": _fallback_zero(agg("sim_o_mean"), "sim_o_mean"),
            "sim_consistency": _fallback_zero(
                agg("sim_consistency"), "sim_consistency"
            ),
            "sim_drift_slope": _fallback_zero(
                agg("sim_drift_slope"), "sim_drift_slope"
            ),
            "confusion_mean": _fallback_zero(agg("confusion_mean"), "confusion_mean"),
            "bleed_db_p50": _fallback_zero(
                _percentile(all_bleed_db, 50), "bleed_db_p50"
            ),
            "bleed_db_p90": _fallback_zero(
                _percentile(all_bleed_db, 90), "bleed_db_p90"
            ),
        }

    # -- per-channel feature extraction ------------------------------------ #
    def _channel_features(
        self, gen_wav_path: Path, prompt_wav_path: Path
    ) -> ChannelFeatures:
        gen_wav, sr = load_wav(gen_wav_path, target_sr=self.embed_sample_rate)
        raw_segments = self.vad(gen_wav, sr)
        ipus = build_ipus(
            raw_segments,
            min_silence=self.min_silence,
            min_speech=self.min_speech,
            pad=self.pad,
            total_duration=len(gen_wav) / sr if sr else None,
        )

        concat_chunks = [
            gen_wav[_sec_to_sample(s, sr) : _sec_to_sample(e, sr)] for s, e in ipus
        ]
        concat_audio = (
            np.concatenate(concat_chunks)
            if concat_chunks
            else np.zeros(0, dtype=np.float32)
        )
        concat_duration = len(concat_audio) / sr if sr else 0.0
        gen_embedding = (
            self._embed(concat_audio, sr)
            if concat_duration >= self.embed_min_sec
            else None
        )

        prompt_wav, psr = load_wav(prompt_wav_path, target_sr=self.embed_sample_rate)
        prompt_duration = len(prompt_wav) / psr if psr else 0.0
        prompt_embedding = (
            self._embed(prompt_wav, psr)
            if prompt_duration >= self.embed_min_sec
            else None
        )

        per_ipu_embeddings: List[Optional[np.ndarray]] = []
        for s, e in ipus:
            if e - s < self.embed_min_sec:
                per_ipu_embeddings.append(None)
                continue
            snippet = gen_wav[_sec_to_sample(s, sr) : _sec_to_sample(e, sr)]
            per_ipu_embeddings.append(self._embed(snippet, sr))

        return ChannelFeatures(
            ipus=ipus,
            gen_embedding=gen_embedding,
            prompt_embedding=prompt_embedding,
            per_ipu_embeddings=per_ipu_embeddings,
        )

    def _embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return np.asarray(self.embedder(wav, sr), dtype=np.float64)
