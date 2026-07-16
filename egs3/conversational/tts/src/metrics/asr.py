"""``ConversationASRMetric``: corpus-level WER, the ASR leg of the lean
measure-stage metric battery (see PLAN-step4.md's 2026-07-15 revision).

Per window (iterating ``meta.scp``), two WER variants are computed, BOTH
corpus-level (pooled substitution/deletion/insertion/hit counts across every
utterance in the run, then ``WER = (S + D + I) / (S + D + H)`` computed ONCE
at summary time). Per-utterance WERs are never averaged -- a mean-of-WERs
would let a run full of tiny reference utterances (a channel with one short
turn) dominate the summary just as much as a run of long ones, whereas
pooling weights every reference WORD equally, the standard corpus-WER
convention.

1. **Per-channel WER** (``wer_channel``): one channel of one window is one
   utterance. Hypothesis = a single whole-file transcription of
   ``channels[ch].gen_wav`` (the ENTIRE generated channel, per the reworked
   infer stage -- there is no more prompt/generated split within the wav).
   Reference = ``channels[ch].ref_text`` (all of that channel's window-turn
   texts, already space-joined by the infer stage). Counts are pooled over
   every channel of every window.
2. **Mixed-channel WER** (``wer_mix``): one whole window is one utterance.
   Hypothesis = a single whole-file transcription of the window's
   ``mix_wav``. Reference = ALL of the window's turns (``meta["turns"]``,
   across every channel) joined in START-TIME order (ties keep the list's
   existing order -- a stable sort, never re-decided per key collision).
   Counts are pooled over windows.

Transcription is ONE backend call per file (``vad_filter=True`` on the real
faster-whisper backend), not the old per-IPU-segment loop: channels and even
the mixdown contain long silences (a channel's own silence during other
speakers' turns; the mixdown's silence during conversational gaps), and
faster-whisper's own internal VAD filtering is what now guards against
whisper's well-known silence-hallucination failure mode, replacing the
manual VAD/IPU segmentation this recipe used to do by hand.

The injected ``normalizer`` (default: whisper's ``EnglishTextNormalizer``)
is applied to BOTH the hypothesis and reference text of every utterance
(channel or mix) before counting -- normalization must be symmetric, or the
counts would silently penalize formatting differences ("25" vs "twenty
five") that have nothing to do with transcription accuracy.

Summary keys (exactly two, both pooled per the corpus convention above, both
``summary_value``-guarded: a run with zero utterances -- or, degenerately, a
zero-word pooled reference -- leaves the key ``None``, never a fabricated
0.0):

* ``wer_channel`` -- pooled per-channel WER.
* ``wer_mix``     -- pooled mixed-channel WER.

Per-window JSONL rows carry the per-channel and mix hypotheses plus their
raw count 4-tuples and a per-window (NOT corpus-pooled) WER for each --
debug-only detail; the run summary always comes from the pooled counts
above, never from averaging these per-window numbers.

Deferred to a later PR (see README.md's "Deferred to the next PR" list):
cpWER channel-permutation search, the ``swap`` flag, and script-following
(turn-order accuracy / Kendall tau / turn-count ratio, and the word-timestamp
machinery that fed it) -- all cut in the 2026-07-15 PR #10 review to keep
this battery lean and easy to review.

Backends are constructor-injectable; the real default is lazy so importing
this module (or constructing the metric) never touches the network or loads
a model:

* ``transcriber``: default :class:`FasterWhisperTranscriber` (faster-whisper
  large-v3, ``vad_filter=True``); the underlying package is imported inside
  the first call, never at module scope or in ``__init__``.
* ``normalizer``: default :class:`WhisperEnglishNormalizer`, soft-importing
  ``whisper.normalizers.EnglishTextNormalizer`` on first call. If
  ``openai-whisper`` is not installed at real-runtime, it RAISES with an
  install hint rather than silently falling back to a weaker normalizer --
  a silent fallback would corrupt cross-run WER comparability.

``jiwer`` (word alignment / WER counts) is imported at module scope: unlike
the transcriber/normalizer this is a lightweight, model-free, network-free
pure-Python/numeric library already present in the eval environment, so
there is nothing to defer.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import jiwer
import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import load_wav, summary_value

Transcriber = Callable[[np.ndarray, int], str]
Normalizer = Callable[[str], str]


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class FasterWhisperTranscriber:
    """Real default transcriber: faster-whisper ``large-v3``, transcribed in
    ONE call per file with ``vad_filter=True`` (faster-whisper's own internal
    VAD filtering -- the anti-silence-hallucination defense for this lean
    battery, replacing the old manual per-IPU segmentation).

    ``faster_whisper`` is imported inside :meth:`_load`, invoked from the
    first :meth:`__call__`, never at module scope or in ``__init__`` --
    constructing this class (e.g. as a metric's default) is always safe
    offline.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "en",
        **model_kwargs: Any,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model_kwargs = model_kwargs
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # network/asset fetch, deferred

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            **self.model_kwargs,
        )

    def __call__(self, wav: np.ndarray, sr: int) -> str:
        self._load()
        segments_iter, _info = self._model.transcribe(
            wav, language=self.language, vad_filter=True
        )
        return " ".join(
            seg.text.strip() for seg in segments_iter if seg.text.strip()
        )


class WhisperEnglishNormalizer:
    """Real default normalizer: whisper's ``EnglishTextNormalizer``.

    Soft-imported inside :meth:`_load` on first call. If ``openai-whisper``
    is not installed, raises with an install hint -- NEVER falls back to a
    weaker normalizer, since that would silently corrupt cross-run WER
    comparability (the whole point of normalizing is that every run uses the
    exact same rules).
    """

    def __init__(self) -> None:
        self._normalizer = None

    def _load(self) -> None:
        if self._normalizer is not None:
            return
        try:
            from whisper.normalizers import EnglishTextNormalizer
        except ImportError as exc:
            raise ImportError(
                "ConversationASRMetric's default normalizer requires "
                "openai-whisper (`pip install openai-whisper`) for "
                "whisper.normalizers.EnglishTextNormalizer. Install it, or "
                "inject a `normalizer=` callable explicitly -- silently "
                "falling back to a weaker normalizer would corrupt "
                "cross-run WER comparability."
            ) from exc
        self._normalizer = EnglishTextNormalizer()

    def __call__(self, text: str) -> str:
        self._load()
        return self._normalizer(text)


# --------------------------------------------------------------------------- #
# corpus-level (pooled) WER counting
# --------------------------------------------------------------------------- #
def _counts(ref: str, hyp: str) -> Dict[str, int]:
    """One utterance's substitution/deletion/insertion/hit counts."""
    out = jiwer.process_words(ref, hyp)
    return {
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
    }


def _wer_from_counts(counts: Dict[str, int]) -> Optional[float]:
    """``(S + D + I) / (S + D + H)``; ``None`` (never a fabricated 0.0) when
    the denominator (total reference words) is zero."""
    denom = counts["substitutions"] + counts["deletions"] + counts["hits"]
    if denom == 0:
        return None
    numer = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return numer / denom


def _pool_wer(counts_list: Sequence[Dict[str, int]]) -> Optional[float]:
    """Pool a sequence of per-utterance count dicts into ONE corpus-level
    WER -- the core "never average per-utterance WERs" primitive, kept as a
    pure function so it is directly unit-testable independent of any audio
    I/O or file fixture."""
    pooled = {
        "substitutions": sum(c["substitutions"] for c in counts_list),
        "deletions": sum(c["deletions"] for c in counts_list),
        "insertions": sum(c["insertions"] for c in counts_list),
        "hits": sum(c["hits"] for c in counts_list),
    }
    return _wer_from_counts(pooled)


def _mix_reference(turns: Sequence[Dict[str, Any]]) -> str:
    """All of a window's turns, joined in START-TIME order (a stable sort,
    so turns that tie on ``start`` keep their existing list order)."""
    ordered = sorted(turns, key=lambda t: t["start"])
    return " ".join(t["text"] for t in ordered)


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class ConversationASRMetric(BaseMetric):
    """ASR leg of the lean measure-stage battery: corpus-level per-channel
    and mixed-channel WER. See module docstring."""

    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        normalizer: Optional[Normalizer] = None,
        asr_sample_rate: int = 16000,
    ) -> None:
        self.transcriber = (
            transcriber if transcriber is not None else FasterWhisperTranscriber()
        )
        self.normalizer = (
            normalizer if normalizer is not None else WhisperEnglishNormalizer()
        )
        self.asr_sample_rate = asr_sample_rate

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "conversation_asr"
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
        turns = meta.get("turns", [])

        channel_records = []
        for ch in channels_meta:
            hyp_raw = self._transcribe(test_dir / ch["gen_wav"])
            ref_raw = ch.get("ref_text", "")
            hyp_norm = self.normalizer(hyp_raw)
            ref_norm = self.normalizer(ref_raw)
            counts = _counts(ref_norm, hyp_norm)
            channel_records.append(
                {
                    "hyp_text": hyp_raw,
                    "ref_text": ref_raw,
                    "counts": counts,
                    "wer": _wer_from_counts(counts),
                }
            )

        mix_hyp_raw = self._transcribe(test_dir / meta["mix_wav"])
        mix_ref_raw = _mix_reference(turns)
        mix_hyp_norm = self.normalizer(mix_hyp_raw)
        mix_ref_norm = self.normalizer(mix_ref_raw)
        mix_counts = _counts(mix_ref_norm, mix_hyp_norm)

        return {
            "window_id": window_id,
            "num_channels": len(channels_meta),
            "channels": channel_records,
            "mix": {
                "hyp_text": mix_hyp_raw,
                "ref_text": mix_ref_raw,
                "counts": mix_counts,
                "wer": _wer_from_counts(mix_counts),
            },
        }

    def _summarize(
        self, per_window: Sequence[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        name = type(self).__name__
        channel_counts = [c["counts"] for w in per_window for c in w["channels"]]
        mix_counts = [w["mix"]["counts"] for w in per_window]
        return {
            "wer_channel": summary_value(
                _pool_wer(channel_counts), "wer_channel", metric_name=name
            ),
            "wer_mix": summary_value(
                _pool_wer(mix_counts), "wer_mix", metric_name=name
            ),
        }

    # -- transcription ------------------------------------------------------ #
    def _transcribe(self, wav_path: Path) -> str:
        wav, sr = load_wav(wav_path, target_sr=self.asr_sample_rate)
        return self.transcriber(wav, sr)
