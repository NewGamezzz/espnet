"""``ConversationASRMetric``: the ASR leg of the measure-stage metric battery.

Per window (iterating ``meta.scp``), for every generated channel:

1. Segment the channel's generated-region wav into IPUs (``segments.py``,
   dGSLM 200 ms rule) and transcribe EACH IPU separately (rather than the
   whole channel in one call) -- this is what avoids whisper's well-known
   silence-hallucination failure mode on long quiet stretches. Word
   timestamps returned by the transcriber are IPU-local; this module offsets
   them by the IPU's start so all of a channel's words share one timeline
   (seconds from the start of its generated region). The channel hypothesis
   text is the space-joined concatenation of its IPU transcripts, in IPU
   order (IPUs are already time-sorted by ``build_ipus``).
2. Per-channel WER: the channel hypothesis vs. that channel's ``ref_text``
   from the meta JSON (identity channel<->speaker pairing, since this
   recipe's channels ARE speakers), with the injected ``normalizer`` applied
   to both sides.
3. cpWER: the same per-channel hypotheses vs. the per-channel references,
   minimized over every channel<->speaker assignment (``N <= 4`` in this
   recipe, so brute-force permutation is cheap); ``swap`` is flagged when
   the minimizing assignment is not the identity. Ties prefer the identity
   assignment (documented convention, see ``_cpwer``).
4. Script following: EACH channel's hypothesis WORDS are aligned (per
   channel, via jiwer's edit-distance alignment) to THAT channel's scripted
   turn texts (generated-region turns only, i.e. turns with
   ``start >= prompt_boundary_sec``, in script order); a turn's "realized
   time" is the mean midpoint timestamp of the hypothesis words the
   alignment attributes to it. The per-channel realized times are then
   scattered back into ONE pooled, cross-channel timeline in the window's
   original script order (``meta["turns"]`` is already chronological across
   ALL channels, and every channel's generated region starts at the SAME
   ``prompt_boundary_sec`` instant, so per-channel realized times share one
   clock and are directly comparable). Turn-order accuracy, Kendall tau, and
   turn-count ratio are computed ONCE per window over this pooled,
   cross-channel sequence -- this is what actually measures conversational
   turn-taking order (whether speaker A's and B's turns interleaved
   correctly), not merely whether each speaker's own turns stayed in order
   (which they trivially do, since a single speaker's turns never overlap
   themselves).

Summary keys (each is a MEAN OVER WINDOWS of a per-window scalar):

* ``wer_ch_mean``   -- per-window mean of per-channel WER, then mean over windows.
* ``wer_ch_worst``  -- per-window max of per-channel WER, then mean over windows.
* ``cpwer``         -- per-window cpWER (already a single global number per
  window, computed over ALL channels jointly), meaned over windows.
* ``swap_rate``     -- fraction of windows whose cpWER-minimizing assignment
  was not the identity.
* ``turn_order_acc``, ``kendall_tau``, ``turn_count_ratio`` -- the pooled,
  cross-channel per-window scalar described above, meaned over windows that
  had at least one generated-region turn (``turn_order_acc``/``kendall_tau``
  further require at least one / two REALIZED turns respectively; windows
  without a defined value are skipped, not counted as 0).

Two DISTINCT normalization steps are used, deliberately not shared:

* The injected ``normalizer`` (default: whisper's ``EnglishTextNormalizer``)
  runs on whole reference/hypothesis STRINGS for WER/cpWER (item 2/3 above).
  It may merge or split tokens (e.g. "twenty five" -> "25"), which is fine
  for a scalar error rate but would silently break the word<->timestamp
  mapping script-following depends on.
* Script-following (item 4) instead normalizes each WORD independently
  (lowercase, strip surrounding punctuation, see ``_normalize_word``) so
  every hypothesis word keeps exactly its own transcriber timestamp.

Missing-turn convention (documented per the task's explicit ask): a scripted
turn with zero aligned hypothesis words is "missing". Missing turns count
against ``turn_count_ratio`` (realized / scripted) but are EXCLUDED from
``turn_order_acc`` and ``kendall_tau``, which are computed over the realized
turns only (in their original script order) -- there is no realized time to
rank a missing turn by. ``turn_order_acc`` is the fraction of realized
turns (in the pooled, cross-channel sequence) whose rank-by-realized-time
matches their rank-by-script-order among the OTHER realized turns of that
window (i.e. does time-sorting the realized turns recover the identity
permutation); ``kendall_tau`` is the Kendall rank-correlation (scipy)
between script order and realized-time order over the same set, requiring
at least 2 realized turns.

Backends are constructor-injectable; real defaults are lazy so importing
this module (or constructing the metric) never touches the network or loads
a model:

* ``transcriber``: default :class:`FasterWhisperTranscriber` (faster-whisper
  large-v3); the underlying package is imported inside the first call, never
  at module scope or in ``__init__``.
* ``normalizer``: default :class:`WhisperEnglishNormalizer`, soft-importing
  ``whisper.normalizers.EnglishTextNormalizer`` on first call. If
  ``openai-whisper`` is not installed at real-runtime, it RAISES with an
  install hint rather than silently falling back to a weaker normalizer --
  a silent fallback would corrupt cross-run WER comparability.
* ``vad``: default ``segments.VAD()`` (lazy silero), per the shared Task-2
  utility.

``jiwer`` (word alignment / WER) and ``scipy`` (Kendall tau) are imported at
module scope: unlike the transcriber/normalizer these are lightweight,
model-free, network-free pure-Python/numeric libraries already present in
the eval environment, so there is nothing to defer.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jiwer
import numpy as np
from scipy.stats import kendalltau

from espnet3.components.metrics.base_metric import BaseMetric

from .segments import VAD, Interval, build_ipus, load_wav

logger = logging.getLogger(__name__)

_TURN_EPS = 1e-6


# --------------------------------------------------------------------------- #
# word type + backends
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Word:
    """One transcribed word. ``start``/``end`` are seconds, relative to
    whatever audio snippet the transcriber was called on (IPU-local until
    :meth:`ConversationASRMetric._transcribe_channel` offsets them)."""

    text: str
    start: float
    end: float


Transcriber = Callable[[np.ndarray, int], Sequence[Word]]
Normalizer = Callable[[str], str]


class FasterWhisperTranscriber:
    """Real default transcriber: faster-whisper ``large-v3``.

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

    def __call__(self, wav: np.ndarray, sr: int) -> List[Word]:
        self._load()
        segments_iter, _info = self._model.transcribe(
            wav, language=self.language, word_timestamps=True
        )
        words: List[Word] = []
        for segment in segments_iter:
            for w in segment.words or []:
                text = w.word.strip()
                if text:
                    words.append(
                        Word(text=text, start=float(w.start), end=float(w.end))
                    )
        return words


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
# small helpers
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _mean_skip_none(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fallback_zero(value: Optional[float], key: str) -> float:
    if value is None:
        logger.warning(
            "ConversationASRMetric: no window produced a defined value for "
            "'%s'; defaulting the run summary to 0.0",
            key,
        )
        return 0.0
    return float(value)


_WORD_STRIP_RE = re.compile(r"^[^\w']+|[^\w']+$")


def _normalize_word(text: str) -> str:
    """Lowercase + strip surrounding punctuation for ONE word.

    Used only for script-following alignment (see module docstring for why
    this is separate from the injected sentence-level ``normalizer``).
    """
    return _WORD_STRIP_RE.sub("", text.lower())


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
class ConversationASRMetric(BaseMetric):
    """ASR leg of the measure-stage battery: WER, cpWER, script following."""

    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        normalizer: Optional[Normalizer] = None,
        vad: Optional[Callable[[np.ndarray, int], Sequence[Interval]]] = None,
        asr_sample_rate: int = 16000,
        min_silence: float = 0.2,
        min_speech: float = 0.0,
        pad: float = 0.1,
    ) -> None:
        self.transcriber = (
            transcriber if transcriber is not None else FasterWhisperTranscriber()
        )
        self.normalizer = (
            normalizer if normalizer is not None else WhisperEnglishNormalizer()
        )
        self.vad = vad if vad is not None else VAD()
        self.asr_sample_rate = asr_sample_rate
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.pad = pad

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
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
        boundary_sec = float(meta["prompt_boundary_sec"])
        channels_meta = meta["channels"]
        turns = meta.get("turns", [])
        n = len(channels_meta)

        hyp_words: List[List[Word]] = []
        hyp_texts: List[str] = []
        ref_texts: List[str] = [ch.get("ref_text", "") for ch in channels_meta]

        for ch in channels_meta:
            words = self._transcribe_channel(test_dir / ch["gen_wav"])
            hyp_words.append(words)
            hyp_texts.append(" ".join(w.text for w in words))

        ref_norm = [self.normalizer(t) for t in ref_texts]
        hyp_norm = [self.normalizer(t) for t in hyp_texts]

        channel_wer = [float(jiwer.wer(r, h)) for r, h in zip(ref_norm, hyp_norm)]
        cpwer, swap, assignment = self._cpwer(ref_norm, hyp_norm)

        # Pooled, cross-channel script following: per-channel alignment
        # produces a realized time per turn, but the order/tau/ratio stats
        # are computed ONCE over the window's whole chronological turn
        # sequence (see module docstring for why -- this is what actually
        # measures conversational turn-taking order across speakers).
        gen_turns = [t for t in turns if t["start"] >= boundary_sec - _TURN_EPS]
        channel_positions: List[List[int]] = [[] for _ in range(n)]
        for gi, t in enumerate(gen_turns):
            channel_positions[t["channel"]].append(gi)

        global_realized_time: List[Optional[float]] = [None] * len(gen_turns)
        channel_debug: List[Dict[str, Any]] = []
        for k in range(n):
            positions = channel_positions[k]
            local_texts = [gen_turns[gi]["text"] for gi in positions]
            local_times = self._align_words_to_turns(local_texts, hyp_words[k])
            for local_idx, gi in enumerate(positions):
                global_realized_time[gi] = local_times[local_idx]
            channel_debug.append(
                {"scripted_turns": local_texts, "realized_times": local_times}
            )

        script_stats = self._script_order_stats(global_realized_time)

        return {
            "window_id": window_id,
            "num_channels": n,
            "wer_ch_mean": _mean(channel_wer),
            "wer_ch_worst": max(channel_wer) if channel_wer else None,
            "cpwer": cpwer,
            "swap": swap,
            "cpwer_assignment": list(assignment),
            "turn_order_acc": script_stats["turn_order_acc"],
            "kendall_tau": script_stats["kendall_tau"],
            "turn_count_ratio": script_stats["turn_count_ratio"],
            "channels": [
                {
                    "hyp_text": hyp_texts[k],
                    "ref_text": ref_texts[k],
                    "wer": channel_wer[k],
                    **channel_debug[k],
                }
                for k in range(n)
            ],
            "script_following": {
                "turn_channels": [t["channel"] for t in gen_turns],
                "realized_times": global_realized_time,
                "missing_turns": script_stats["missing_turns"],
                "num_scripted_turns": script_stats["num_scripted_turns"],
                "num_realized_turns": script_stats["num_realized_turns"],
            },
        }

    def _summarize(self, per_window: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        def agg(key: str) -> Optional[float]:
            return _mean_skip_none(w[key] for w in per_window)

        swap_rate = _mean_skip_none(1.0 if w["swap"] else 0.0 for w in per_window)
        return {
            "wer_ch_mean": _fallback_zero(agg("wer_ch_mean"), "wer_ch_mean"),
            "wer_ch_worst": _fallback_zero(agg("wer_ch_worst"), "wer_ch_worst"),
            "cpwer": _fallback_zero(agg("cpwer"), "cpwer"),
            "swap_rate": _fallback_zero(swap_rate, "swap_rate"),
            "turn_order_acc": _fallback_zero(agg("turn_order_acc"), "turn_order_acc"),
            "kendall_tau": _fallback_zero(agg("kendall_tau"), "kendall_tau"),
            "turn_count_ratio": _fallback_zero(
                agg("turn_count_ratio"), "turn_count_ratio"
            ),
        }

    # -- transcription ------------------------------------------------------ #
    def _transcribe_channel(self, wav_path: Path) -> List[Word]:
        wav, sr = load_wav(wav_path, target_sr=self.asr_sample_rate)
        raw_segments = self.vad(wav, sr)
        ipus = build_ipus(
            raw_segments,
            min_silence=self.min_silence,
            min_speech=self.min_speech,
            pad=self.pad,
            total_duration=len(wav) / sr if sr else None,
        )
        words: List[Word] = []
        for start, end in ipus:
            s_samp = max(0, int(round(start * sr)))
            e_samp = min(len(wav), int(round(end * sr)))
            if e_samp <= s_samp:
                continue
            snippet = wav[s_samp:e_samp]
            for w in self.transcriber(snippet, sr):
                words.append(
                    Word(text=w.text, start=w.start + start, end=w.end + start)
                )
        return words

    # -- WER / cpWER ---------------------------------------------------- #
    @staticmethod
    def _cpwer(
        ref_norm: Sequence[str], hyp_norm: Sequence[str]
    ) -> Tuple[float, bool, Tuple[int, ...]]:
        """Min-WER channel<->speaker assignment (brute force; N <= 4 here).

        Ties prefer the identity assignment: ``itertools.permutations`` on
        ``range(n)`` yields the identity tuple first, and ``min`` keeps the
        first minimizer it sees, so a tie never spuriously flags ``swap``.
        """
        n = len(ref_norm)
        identity = tuple(range(n))
        if n == 0:
            return 0.0, False, identity

        best_perm, best_wer = identity, None
        for perm in itertools.permutations(range(n)):
            refs = [ref_norm[j] for j in perm]
            out = jiwer.process_words(refs, list(hyp_norm))
            if best_wer is None or out.wer < best_wer:
                best_wer, best_perm = out.wer, perm
        return float(best_wer), best_perm != identity, best_perm

    # -- script following ------------------------------------------------ #
    @staticmethod
    def _align_words_to_turns(
        scripted_texts: Sequence[str], words: Sequence[Word]
    ) -> List[Optional[float]]:
        """Per-CHANNEL alignment core: one realized time per scripted turn.

        Aligns this channel's hypothesis words to this channel's own
        scripted turn texts (edit-distance, jiwer) and returns, per turn,
        the mean midpoint timestamp of the hypothesis words attributed to
        it (``None`` when no word aligned -- a "missing" turn). Pure
        per-channel bookkeeping; cross-channel order/tau/ratio stats are
        computed separately by :meth:`_script_order_stats` over the POOLED
        sequence (see module docstring).
        """
        num_scripted = len(scripted_texts)
        if num_scripted == 0:
            return []

        ref_words: List[str] = []
        turn_of_word: List[int] = []
        for turn_idx, text in enumerate(scripted_texts):
            for tok in text.split():
                nw = _normalize_word(tok)
                if nw:
                    ref_words.append(nw)
                    turn_of_word.append(turn_idx)

        hyp_pairs: List[Tuple[str, Word]] = []
        for w in words:
            nw = _normalize_word(w.text)
            if nw:
                hyp_pairs.append((nw, w))

        turn_times: List[List[float]] = [[] for _ in range(num_scripted)]
        if ref_words and hyp_pairs:
            out = jiwer.process_words(
                " ".join(ref_words), " ".join(p[0] for p in hyp_pairs)
            )
            for chunk in out.alignments[0]:
                if chunk.type not in ("equal", "substitute"):
                    continue
                ref_idxs = range(chunk.ref_start_idx, chunk.ref_end_idx)
                hyp_idxs = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
                # Substitute blocks are not guaranteed equal-length on both
                # sides; pair up positionally and leave any excess
                # unattributed (same spirit as an insertion/deletion).
                for r_i, h_i in zip(ref_idxs, hyp_idxs):
                    turn_idx = turn_of_word[r_i]
                    _, word_obj = hyp_pairs[h_i]
                    turn_times[turn_idx].append((word_obj.start + word_obj.end) / 2.0)

        return [_mean(times) if times else None for times in turn_times]

    @staticmethod
    def _script_order_stats(
        realized_time: Sequence[Optional[float]],
    ) -> Dict[str, Any]:
        """Window-level, cross-channel order/tau/ratio stats.

        ``realized_time`` is indexed by POOLED script order (chronological
        across all channels, per ``meta["turns"]``); see module docstring
        for the missing-turn convention and the exact ``turn_order_acc`` /
        ``kendall_tau`` definitions.
        """
        num_scripted = len(realized_time)
        if num_scripted == 0:
            return {
                "num_scripted_turns": 0,
                "num_realized_turns": 0,
                "missing_turns": [],
                "turn_order_acc": None,
                "kendall_tau": None,
                "turn_count_ratio": None,
            }

        realized_indices = [
            i for i in range(num_scripted) if realized_time[i] is not None
        ]
        missing = [i for i in range(num_scripted) if realized_time[i] is None]

        if realized_indices:
            actual_order = sorted(realized_indices, key=lambda i: realized_time[i])
            matches = sum(
                1
                for expected, actual in zip(realized_indices, actual_order)
                if expected == actual
            )
            turn_order_acc = matches / len(realized_indices)
        else:
            turn_order_acc = None

        if len(realized_indices) >= 2:
            x = list(range(len(realized_indices)))
            y = [realized_time[i] for i in realized_indices]
            tau = kendalltau(x, y).statistic
            kendall_tau = None if (tau is None or math.isnan(tau)) else float(tau)
        else:
            kendall_tau = None

        return {
            "num_scripted_turns": num_scripted,
            "num_realized_turns": len(realized_indices),
            "missing_turns": missing,
            "turn_order_acc": turn_order_acc,
            "kendall_tau": kendall_tau,
            "turn_count_ratio": len(realized_indices) / num_scripted,
        }
