"""CoVoMix2 dialogue test-set adapter for the conversational multi-branch F5
recipe.

Loads the public CoVoMix2 evaluation stimuli
(https://github.com/vivian556123/covomix2-dialogue-testset) -- 1000 written
DailyDialog dialogues, each paired with two LibriSpeech ``test-clean``
acoustic prompts -- into records the ``infer`` stage can generate from.

Nothing here reads or writes the SSSD manifests, and nothing here is
imported by the SSSD ``infer`` path or by training: the existing
``src/inference.py`` numbers stay bit-reproducible.

Why this set needs its own adapter
----------------------------------
The SSSD path derives the generated window's DURATION from the ground-truth
window it is reconstructing (``src/inference.py`` sets ``total_frames`` from
``prompt + window_speech``).  This test set ships no dialogue audio, so the
duration must be PREDICTED -- and duration is the only timing signal this
model ever receives (the masking scheme is turn-order only: one ``<turn>``
per turn, one ``<OTHER>`` per character of the other speaker's text, never a
timestamp).  Duration estimation is therefore a first-class, reported
hyperparameter here, not an implementation detail; see
:func:`estimate_duration_sec`.

Test-set layout consumed
------------------------
* ``dailydialog-dialogue.json`` - a list of entries, each with ``key``,
  ``text`` (a repo-relative path to the transcript), and, per speaker,
  ``audio_prompt_spk{1,2}`` (a LibriSpeech-root-relative FLAC path) plus
  ``audio_prompt_spk{1,2}_transcription``.
* ``transcriptions/<key>.txt`` - one turn per line, blank lines ignored,
  speakers STRICTLY ALTERNATING starting at speaker 1.  That alternation is
  the test set's own convention; it is asserted rather than inferred, since
  a silent mis-assignment would swap every reference text.

Text handling
-------------
DailyDialog text is raw, so every turn and every prompt transcription goes
through the SAME :func:`~egs3.conversational.tts.dataset.preprocessing.text
.normalize_text` the SSSD builder applies at build time (F5's own tokenizer
plus charset filtering).  Skipping it would hand the preprocessor characters
outside the extended vocab, and ``encode_tokens`` fails loudly on OOV by
design.

Turn times
----------
Records carry ``Turn`` objects whose ``start``/``end`` are ORDINAL indices,
not seconds -- there is no reference audio to take times from.  The measure
stage only ever uses them to recover conversation order for the mixed-channel
WER reference (``sorted(turns, key=lambda t: t["start"])``, a stable sort),
which ordinals give exactly.  ``meta["turn_times"] == "ordinal"`` marks this
in the output so no downstream reader mistakes them for seconds.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    normalize_text,
    vocab_charset,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab

logger = logging.getLogger(__name__)

DIALOGUE_INDEX = "dailydialog-dialogue.json"

# --------------------------------------------------------------------------- #
# duration-estimation constants
# --------------------------------------------------------------------------- #
# All three are MEASURED, not guessed; each is reported here with the
# population it was measured over so a rebuild can re-derive it.
#
# SSSD_SPEECH_SEC_PER_CHAR - median of sum(channel_speech_sec) / total turn
#   characters over the 49,179 two-speaker windows of the seed-0 SSSD train
#   manifest.  This is an ARTICULATION rate: speech time only, silence
#   excluded.
SSSD_SPEECH_SEC_PER_CHAR = 0.0735
# LIBRISPEECH_SEC_PER_CHAR - median of utterance duration / transcription
#   length over all 2000 prompt utterances this test set references.  Read
#   speech runs ~6% faster per character than SSSD's spontaneous speech, so
#   the F5 ratio rule (which measures rate on the PROMPT) needs correcting
#   toward the target domain.
LIBRISPEECH_SEC_PER_CHAR = 0.0690
# SSSD_SPEECH_DENSITY - median of sum(channel_speech_sec) / window duration
#   over the same windows.  Below 1.0 means the two speakers' speech time
#   sums to slightly less than the wall-clock window: conversational gaps and
#   overlapping speech very nearly cancel, so the silence allowance the F5
#   rule omits is only a few percent here.
SSSD_SPEECH_DENSITY = 0.954

# Correction applied on top of the prompt-measured rate: retarget read-speech
# articulation to spontaneous, then add the conversational silence budget.
#   (0.0735 / 0.0690) / 0.954 = 1.117
# Cross-check: 0.0690 * 1.117 = 0.0771 s/char, against 0.0776 s/char measured
# DIRECTLY as median(window duration / chars) on the same SSSD windows -- two
# independent routes agreeing to 0.7%.
DEFAULT_DURATION_SCALE = 1.117


@dataclass(frozen=True)
class ExternalPrompt:
    """One speaker's acoustic prompt: an absolute audio path plus its text."""

    channel: int
    audio_path: Path
    text: str


@dataclass(frozen=True)
class ExternalRecord:
    """One dialogue: ordered turns plus one acoustic prompt per channel."""

    dialogue_id: str
    num_channels: int
    turns: list[Turn]
    prompts: list[ExternalPrompt]

    @property
    def channel_chars(self) -> list[int]:
        """Per-channel utf-8 byte count of the normalized turn text.

        utf-8 BYTES, matching F5-TTS's own duration rule
        (``len(ref_text.encode("utf-8"))`` in ``utils_infer.py``); for the
        ASCII-range text this recipe supports that equals the character
        count the ``<OTHER>`` budget is built from.
        """
        counts = [0] * self.num_channels
        for turn in self.turns:
            counts[turn.channel] += len(turn.text.encode("utf-8"))
        return counts


def _read_turns(path: Path, num_channels: int) -> list[Turn]:
    """Parse a transcript file into strictly alternating turns.

    Blank lines are dropped BEFORE alternation is assigned, so a stray blank
    line cannot silently shift every subsequent turn to the wrong speaker.
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"{path}: no non-empty transcript lines")
    return [
        Turn(
            channel=i % num_channels,
            speaker=f"spk{i % num_channels + 1}",
            text=line,
            start=float(i),  # ordinal, NOT seconds (see module docstring)
            end=float(i),
        )
        for i, line in enumerate(lines)
    ]


def load_covomix2_testset(
    testset_root: str | Path,
    librispeech_root: str | Path,
    token_list: str | Path,
    *,
    num_channels: int = 2,
) -> list[ExternalRecord]:
    """Load every dialogue, normalizing all text against the extended vocab.

    ``librispeech_root`` is the directory CONTAINING ``test-clean/`` (the
    index's prompt paths are relative to it).  Every referenced prompt file
    must exist: a missing prompt is a setup error, and silently dropping
    dialogues would change the evaluated subset without saying so.
    """
    testset_root = Path(testset_root)
    librispeech_root = Path(librispeech_root)
    charset = vocab_charset(read_vocab(token_list))

    index_path = testset_root / DIALOGUE_INDEX
    entries = json.loads(index_path.read_text(encoding="utf-8"))

    records: list[ExternalRecord] = []
    for entry in entries:
        key = str(entry["key"])
        turns = _read_turns(testset_root / entry["text"], num_channels)
        turns = [
            Turn(
                channel=t.channel,
                speaker=t.speaker,
                text=normalize_text(t.text, charset),
                start=t.start,
                end=t.end,
            )
            for t in turns
        ]
        empty = [t for t in turns if not t.text]
        if empty:
            raise ValueError(
                f"{key}: {len(empty)} turn(s) normalized to empty text; the "
                "masking scheme cannot represent a zero-character turn"
            )

        prompts = []
        for ch in range(num_channels):
            rel = entry[f"audio_prompt_spk{ch + 1}"]
            audio_path = librispeech_root / rel
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"{key}: prompt audio {audio_path} not found; point "
                    "`librispeech_root` at the directory containing test-clean/"
                )
            text = normalize_text(
                entry[f"audio_prompt_spk{ch + 1}_transcription"], charset
            )
            if not text:
                raise ValueError(f"{key}: speaker {ch + 1} prompt text is empty")
            prompts.append(ExternalPrompt(channel=ch, audio_path=audio_path, text=text))

        records.append(
            ExternalRecord(
                dialogue_id=key,
                num_channels=num_channels,
                turns=turns,
                prompts=prompts,
            )
        )
    return records


def estimate_duration_sec(
    record: ExternalRecord,
    prompt_seconds: Sequence[float],
    *,
    duration_scale: float = DEFAULT_DURATION_SCALE,
    speed: float = 1.0,
) -> float:
    """Predict the generated dialogue's duration, in seconds.

    This is F5-TTS's own duration rule (``utils_infer.py``::

        duration = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len / speed

    i.e. seconds-per-character measured on the reference, scaled by the
    generated text length) generalized to two channels and corrected toward
    the conversational target domain:

    * PER SPEAKER rather than averaged - the two prompts are different
      speakers at different rates, and each speaker's own rate should govern
      their own share of the text.
    * ``duration_scale`` retargets read-speech articulation to spontaneous
      and adds the conversational silence budget (see the constants above).
      Pass ``1.0`` to get F5's uncorrected rule.
    * ``speed`` keeps F5's knob, with F5's sense: larger is faster, so
      shorter.  Sweeping it is how the reported interaction metrics' duration
      sensitivity is measured.

    The prompt region itself is NOT included: unlike F5's single-reference
    case, this recipe's caller already accounts for the prompt frames
    separately when it assembles the conditioning speech.  Returns the
    GENERATED region only.
    """
    if len(prompt_seconds) != record.num_channels:
        raise ValueError(
            f"{record.dialogue_id}: got {len(prompt_seconds)} prompt durations "
            f"for {record.num_channels} channels"
        )
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")

    total = 0.0
    for ch, (prompt_sec, prompt) in enumerate(zip(prompt_seconds, record.prompts)):
        prompt_chars = len(prompt.text.encode("utf-8"))
        if prompt_sec <= 0 or prompt_chars <= 0:
            raise ValueError(
                f"{record.dialogue_id}: channel {ch} prompt is degenerate "
                f"({prompt_sec:.3f}s, {prompt_chars} chars)"
            )
        rate = prompt_sec / prompt_chars  # seconds per character, this speaker
        total += record.channel_chars[ch] * rate
    return total * float(duration_scale) / float(speed)


def assign_shard(
    indices: Sequence[int],
    durations: Sequence[float],
    shard_index: int,
    shard_count: int,
) -> list[int]:
    """Split ``indices`` across ``shard_count`` shards, balanced by length.

    Greedy longest-processing-time: walk the dialogues from longest predicted
    duration to shortest and give each to whichever shard is currently
    lightest.  Ties break on the dialogue index, so the split is fully
    deterministic and every shard can compute it independently - no
    coordination, no shared state, and re-running one shard reproduces
    exactly the same membership.

    Striping (``idx % shard_count``) is NOT good enough here.  Predicted
    duration is heavily skewed - on the CoVoMix2 set the 159 dialogues over
    60 s are 16% of the count but 37% of the audio - so a stripe can hand one
    shard a disproportionate share of the tail and that shard becomes the
    wall clock.

    Balancing on duration is still an APPROXIMATION of cost: attention is
    quadratic in frames, so a long dialogue costs more than its seconds
    suggest and the shard holding the longest ones will finish last.  Good
    enough to keep shards within a modest factor; not a scheduler.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    if shard_count == 1:
        return list(indices)

    loads = [0.0] * shard_count
    buckets: list[list[int]] = [[] for _ in range(shard_count)]
    for idx in sorted(indices, key=lambda i: (-durations[i], i)):
        target = min(range(shard_count), key=lambda s: (loads[s], s))
        buckets[target].append(idx)
        loads[target] += durations[idx]
    return sorted(buckets[shard_index])


def select_records(
    records: Sequence[ExternalRecord],
    durations: Sequence[float],
    selection,
) -> tuple[list[int], dict[str, int]]:
    """Pick the evaluated subset: length band, then optional subsample.

    Returns ``(indices, counts)``.  ``counts`` reports the exclusion reasons
    SEPARATELY (``n_out_of_band`` / ``n_not_sampled``) because they mean
    different things: an out-of-band dialogue could not be generated in this
    model's trained regime, a not-sampled one simply was not asked for.
    Collapsing them into one "skipped" number would read as a mass failure
    whenever a small ``num_dialogues`` is set.

    The length band exists because dialogues whose PREDICTED duration exceeds
    the training ``window_max`` fall outside the regime this model was
    fine-tuned in (and outside F5's own pretraining regime).  Excluded
    dialogues are COUNTED AND LOGGED rather than quietly dropped - a filtered
    subset reported as "the CoVoMix2 test set" would overstate coverage.

    Sharding is NOT applied here: the caller batches the selection first
    (:func:`plan_batches`) and shards over whole batches with
    :func:`assign_shard`, so batch composition - and with it every
    dialogue's noise draw - is identical for every ``shard_count``.
    """
    max_duration = selection.get("max_duration")
    min_duration = selection.get("min_duration")
    eligible = []
    for idx, sec in enumerate(durations):
        if min_duration is not None and sec < float(min_duration):
            continue
        if max_duration is not None and sec > float(max_duration):
            continue
        eligible.append(idx)

    n_out_of_band = len(records) - len(eligible)
    if n_out_of_band:
        logger.info(
            "external test set: %d/%d dialogues outside the predicted-duration "
            "band [%s, %s] s were excluded",
            n_out_of_band,
            len(records),
            min_duration,
            max_duration,
        )

    n_in_band = len(eligible)
    num_dialogues = selection.get("num_dialogues")
    if num_dialogues is not None and len(eligible) > int(num_dialogues):
        rng = random.Random(int(selection.get("seed", 0)))
        eligible = sorted(rng.sample(eligible, int(num_dialogues)))
        logger.info(
            "external test set: subsampled %d of %d in-band dialogues (seed=%s)",
            len(eligible),
            n_in_band,
            selection.get("seed", 0),
        )
    return eligible, {
        "n_out_of_band": n_out_of_band,
        "n_not_sampled": n_in_band - len(eligible),
    }


def plan_batches(
    indices: Sequence[int],
    total_secs: Sequence[float],
    *,
    max_batch_audio_sec: float | None = None,
    max_batch_dialogues: int | None = None,
) -> list[list[int]]:
    """Pack the selected dialogues into ODE batches, deterministically.

    One batch is one ``cfm.sample`` call: every dialogue's channels are rows
    of one packed batch, padded to the batch's longest dialogue.  Packing
    walks the selection from longest ``total_secs`` (prompt + predicted
    generation) to shortest, so batchmates have near-equal lengths and the
    padding waste the budget must price in stays small.

    A batch's cost is ``n_dialogues * longest_member_sec`` - the PADDED audio
    the ODE actually integrates, not the sum of true lengths - and a dialogue
    joins the current batch only while that cost stays within
    ``max_batch_audio_sec`` (and the count within ``max_batch_dialogues``).
    A dialogue longer than the whole budget still gets a singleton batch:
    the budget bounds batch growth, it never excludes work.

    ``None`` budgets mean NO batching (every dialogue is a singleton batch),
    which reproduces the sequential behaviour bit-for-bit - the same
    per-dialogue reseed, the same noise draws.

    Ties break on the dialogue index, so the plan is a pure function of
    (selection, lengths, budget): every shard computes the same plan
    independently, and results are invariant to ``shard_count``.
    """
    if max_batch_audio_sec is not None and max_batch_audio_sec <= 0:
        raise ValueError(f"max_batch_audio_sec must be > 0, got {max_batch_audio_sec}")
    if max_batch_dialogues is not None and max_batch_dialogues < 1:
        raise ValueError(f"max_batch_dialogues must be >= 1, got {max_batch_dialogues}")
    if max_batch_audio_sec is None and max_batch_dialogues is None:
        return [[i] for i in indices]

    ordered = sorted(indices, key=lambda i: (-total_secs[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0.0
    for idx in ordered:
        sec = float(total_secs[idx])
        grown_max = max(current_max, sec)  # == current_max: sorted descending
        if current and (
            (
                max_batch_audio_sec is not None
                and (len(current) + 1) * grown_max > max_batch_audio_sec
            )
            or (
                max_batch_dialogues is not None
                and len(current) + 1 > max_batch_dialogues
            )
        ):
            batches.append(current)
            current, current_max = [], 0.0
        current.append(idx)
        current_max = max(current_max, sec)
    if current:
        batches.append(current)
    return batches


def duration_meta(
    duration_scale: float, speed: float, predicted_sec: float
) -> dict[str, Any]:
    """The duration hyperparameters, recorded per window in ``meta``.

    Written into every meta JSON so a results table can never be read
    without the duration policy that produced it.
    """
    return {
        "predicted_sec": round(float(predicted_sec), 6),
        "duration_scale": float(duration_scale),
        "speed": float(speed),
        "rule": "f5_prompt_ratio_per_speaker",
        "constants": {
            "sssd_speech_sec_per_char": SSSD_SPEECH_SEC_PER_CHAR,
            "librispeech_sec_per_char": LIBRISPEECH_SEC_PER_CHAR,
            "sssd_speech_density": SSSD_SPEECH_DENSITY,
        },
    }
