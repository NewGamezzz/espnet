"""espnet3 ``infer`` stage for the conversational multi-branch F5 recipe.

Batch-generates multi-channel conversations from manifest windows in three
modes that share ONE output layout so the later ``measure`` stage needs zero
metric-side special-casing:

* ``generate`` - run the assembled model (zero-init gates by default);
* ``gt``       - copy the ground-truth window audio;
* ``resynth``  - round-trip the ground truth through mel + Vocos.

Per window the stage builds an acoustic + text PROMPT by sampling one turn
per channel from ELSEWHERE in the same conversation (never from inside the
evaluated window - target leakage is never allowed), concatenating those
turns' audio non-overlapping at the start of the conditioning speech, then
masking a region equal to the FULL ground-truth window.  The model therefore
generates the entire dialog, and every channel is guaranteed a voice
reference in the prompt (a hard project rule: every speaker must have a
prompt reference).

Turn-pool construction (once per manifest, not per window): for a window's
``session_id``, the pool is the union of ``turns`` across every record in the
manifest sharing that ``session_id``, deduplicated by
``(channel, start, end, text)``.

Per-channel candidate selection uses a relaxation ladder (first non-empty
tier wins), picked uniformly with ``random.Random(f"{seed}:{window_id}:{k}")``
so the choice is identical across modes and reruns:

1. non-window AND solo AND duration in ``[prompt.turn_min_sec,
   prompt.turn_max_sec]``;
2. non-window AND solo (duration band dropped);
3. non-window (solo dropped too).

"Non-window" (the turn's span does not overlap the evaluated window's
``[t0, t1)``) is NEVER relaxed.  "Solo" means the turn's span does not
overlap any pool turn of a DIFFERENT channel.  If even tier 3 is empty for
any channel, the window is skipped and counted.

Audio assembly: for each channel ``k`` in ascending order, the FULL
multichannel block of the session file is seek-read over the chosen turn's
``[start, end)`` (same read/resample path as the dataset's window loader,
factored into ``generation.read_audio_span``) and the blocks are
concatenated along time in channel order - so during channel ``k``'s block,
channel ``k`` carries its real speech and every other channel carries its
own real (near-silent, solo-turn) audio, which is maximally training-like
conditioning.  The concatenated prompt is trimmed to a whole number of hops
(remainder dropped from the END) so the prompt/generated boundary is
frame-exact; conditioning speech is ``concat(prompt, window_speech)``.

Text assembly (the default ``text_format: order``): ``sample["turns"] =
[prompt_turn_ch0, prompt_turn_ch1, ...] + window_turns`` (window turns keep
their existing order), then the existing per-branch preprocessor runs
unchanged - prompt turns keep their own ``channel``, and inference uses the
identity channel permutation, so no remapping is needed.

``text_format: timestamps`` (``generate`` only; a ValueError otherwise)
switches that to Mode T over the WHOLE conditioning+target sequence rather
than the target alone: ``prompt_window_layout`` places the concatenated
prompt blocks back-to-back from sequence time 0 and then the window turns at
their GROUND-TRUTH offsets past the prompt (real timestamps - unlike the
chunked path, which has to synthesize a timeline from ordinal turns), and
the preprocessor emits one token per mel frame over all ``total_frames``.
Prompt-turn spans come from the UNTRIMMED block lengths, so they can differ
from the hop-trimmed prompt by at most one hop - the same order of rounding
``turn_frame_spans`` applies to every other boundary.  A window whose turns
do not fit their spans (``timestamp_fits`` is false) DEGRADES to the
order-only text above rather than failing, and is counted in
``n_timestamp_degraded``: one unfittable window must never cost the run.

Output contract, under ``inference_dir/<test_name>/`` (ALL paths in
``meta.scp`` and in the meta JSONs are relative to THIS directory, so the
whole tree is relocatable):

* ``meta.scp``               - ``<window_id> meta/<window_id>.json``; the
  PRIMARY input every metric iterates.
* ``meta/<window_id>.json``  - window duration, sample rate, per-channel
  relative paths (generated / prompt / ground-truth wav) and reference text
  (ALL window turns of that channel), the ground-truth turn spans shifted to
  window time (the generated region now starts at window time 0), the
  prompt's total duration/frames and the concatenated prompt turns
  (session-absolute spans, concatenation order), the mixdown wav path, the
  effective ``text_format`` (present in every mode, ``"order"`` unless the
  window actually ran Mode T, so a degraded window is self-describing) plus
  a ``layout`` block of sequence-time turn spans in Mode T only, and RTF
  (generate mode only).
* convenience SCPs (NOT consumed by metrics): channel-level ``wav.scp`` /
  ``prompt.scp`` / ``text.scp`` (``<window_id>_ch<k>`` rows) and window-level
  ``mix.scp``.

``channels[ch].prompt_wav`` is channel ``ch``'s OWN turn block's channel-``ch``
row only (its solo speech, one turn long) - the speaker-similarity reference
- NOT the whole concatenated prompt region.

Prompt selection and writing happen in ALL THREE modes (the speaker metric
needs prompt references for the gt/resynth anchor runs too); ``gt`` mode
still needs no model/vocoder (pure audio slicing + concatenation), and ``gt``
/ ``resynth`` do not run the text preprocessor.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from egs3.conversational.tts.dataset.preprocessing.text import (
    FRAMES_PER_SECOND,
    timestamp_fits,
)
from egs3.conversational.tts.src.eval_manifest import (
    load_eval_manifest,
    spans_match,
)
from egs3.conversational.tts.src.generation import (
    build_dataset,
    build_preprocessor,
    generate_region,
    load_model,
    load_vocoder,
    pad_branch_text,
    read_audio_span,
    resynth_region,
    write_wav,
)
from egs3.conversational.tts.src.external_testset import (
    DEFAULT_DURATION_SCALE,
    SSSD_SPEECH_SEC_PER_CHAR,
    duration_meta,
)
from egs3.conversational.tts.src.timestamp_layout import prompt_window_layout

logger = logging.getLogger(__name__)

_EPS = 1e-6
_MODES = ("generate", "gt", "resynth")
_TEXT_FORMATS = ("order", "timestamps")
_DURATION_SOURCES = ("ground_truth", "predicted")


def predict_generated_sec(
    prompt_secs,
    prompt_texts,
    channel_texts,
    *,
    duration_scale: float = DEFAULT_DURATION_SCALE,
    speed: float = 1.0,
    rate_prior_chars=None,
    rate_prior_sec_per_char: float = SSSD_SPEECH_SEC_PER_CHAR,
) -> float:
    """F5's duration rule per speaker, as in ``external_testset``
    (``speaker_rates`` + ``estimate_duration_sec``) but on plain lists:
    rate_k = (prompt_sec_k + k * prior) / (prompt_chars_k + k), generated
    seconds = sum_k chars_k * rate_k * scale / speed.  Characters are utf-8
    bytes (F5's own count).  ``rate_prior_chars`` None/0 = the raw prompt
    rate; the prior is a training-set constant, never a test-set number.
    """
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    k = 0.0 if rate_prior_chars is None else float(rate_prior_chars)
    if k < 0:
        raise ValueError(f"rate_prior_chars must be >= 0 or null, got {k}")
    total = 0.0
    for ch, (psec, ptext, ctext) in enumerate(zip(prompt_secs, prompt_texts, channel_texts)):
        pchars = len(ptext.encode("utf-8"))
        if psec <= 0 or pchars <= 0:
            raise ValueError(f"channel {ch} prompt is degenerate ({psec:.3f}s, {pchars} chars)")
        rate = (float(psec) + k * float(rate_prior_sec_per_char)) / (pchars + k)
        total += len(ctext.encode("utf-8")) * rate
    return total * float(duration_scale) / float(speed)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True iff the half-open spans ``[a_start, a_end)``/``[b_start, b_end)`` overlap."""
    return a_start < b_end and b_start < a_end


def _build_turn_pools(records) -> dict[str, list[Any]]:
    """Per-session turn pools: union of ``turns`` across records sharing a
    ``session_id``, deduplicated by ``(channel, start, end, text)``, in
    first-seen order (deterministic regardless of hash randomization since
    membership uses a set but output order only ever follows record/turn
    iteration order).  Built once for the whole manifest, not per window.
    """
    pools: dict[str, list[Any]] = {}
    seen: dict[str, set[tuple]] = {}
    for record in records:
        pool = pools.setdefault(record.session_id, [])
        seen_keys = seen.setdefault(record.session_id, set())
        for turn in record.turns:
            key = (turn.channel, turn.start, turn.end, turn.text)
            if key not in seen_keys:
                seen_keys.add(key)
                pool.append(turn)
    return pools


def load_excluded_spans(path) -> dict[str, frozenset[tuple[int, float, float]]]:
    """``prompt.exclude_spans`` sidecar (written by local/ami_prompt_gate.py):
    turns that may never serve as a prompt, keyed by session.  Applied to
    every ladder tier - a gated turn is not a candidate at all."""
    data = json.loads(Path(path).read_text("utf-8"))
    if data.get("version") != 1:
        raise ValueError(
            f"{path}: unsupported exclude_spans version {data.get('version')!r}"
        )
    out: dict[str, set] = {}
    for s in data["spans"]:
        out.setdefault(str(s["session_id"]), set()).add(
            (int(s["channel"]), round(float(s["start"]), 6), round(float(s["end"]), 6))
        )
    return {k: frozenset(v) for k, v in out.items()}


def _select_prompt_turn(
    pool_turns: Sequence[Any],
    channel: int,
    t0: float,
    t1: float,
    turn_min: float,
    turn_max: float,
    seed: Any,
    window_id: str,
    *,
    solo_guard: float = 0.0,
    excluded: frozenset = frozenset(),
):
    """Pick one prompt turn for ``channel`` via the relaxation ladder (module
    docstring).  Returns ``None`` iff even the loosest tier (non-window) is
    empty - target leakage is never allowed, so that tier is never relaxed.

    ``channel`` is a SOURCE channel (a headset index for AMI; identical to
    the row index when the window uses every column).  ``solo_guard`` widens
    the candidate span by that many seconds per side before the solo test;
    ``excluded`` (``(channel, start, end)`` keys, see ``load_excluded_spans``)
    removes turns from every tier.
    """
    non_window = [
        t
        for t in pool_turns
        if t.channel == channel
        and not _overlaps(t.start, t.end, t0, t1)
        and (t.channel, round(t.start, 6), round(t.end, 6)) not in excluded
    ]
    if not non_window:
        return None

    def _is_solo(turn) -> bool:
        return not any(
            _overlaps(turn.start - solo_guard, turn.end + solo_guard, other.start, other.end)
            for other in pool_turns
            if other.channel != channel
        )

    solo = [t for t in non_window if _is_solo(t)]
    banded = [
        t for t in solo if turn_min - _EPS <= (t.end - t.start) <= turn_max + _EPS
    ]
    for tier in (banded, solo, non_window):
        if tier:
            rng = random.Random(f"{seed}:{window_id}:{channel}")
            return rng.choice(tier)
    return None  # unreachable: non_window already checked non-empty


def _in_duration_band(duration: float, min_duration, max_duration) -> bool:
    if min_duration is not None and duration < float(min_duration):
        return False
    if max_duration is not None and duration > float(max_duration):
        return False
    return True


def _select_indices(records, selection, pinned_rows=None) -> list[int]:
    """Filtered, seeded, capped window indices (sorted for determinism).

    ``pinned_rows`` (a frozen eval manifest's window rows) supersedes the
    whole draw: filter, cap and seed are all ignored and the listed windows
    run in MANIFEST ORDER.  Each row's ``session_id`` / ``t0`` / ``t1`` are
    checked against the split, so a manifest built against different data
    raises instead of silently scoring different windows.
    """
    if pinned_rows is not None:
        by_id = {r.window_id: i for i, r in enumerate(records)}
        indices = []
        for row in pinned_rows:
            wid = row["window_id"]
            if wid not in by_id:
                raise ValueError(
                    f"eval manifest names window {wid!r}, which is not in "
                    f"this split ({len(records)} windows)"
                )
            idx = by_id[wid]
            record = records[idx]
            for key, actual in (("t0", record.t0), ("t1", record.t1)):
                if key in row and not spans_match(row[key], actual):
                    raise ValueError(
                        f"eval manifest {wid}: {key} is {row[key]} but the "
                        f"split says {actual} - the manifest was built "
                        f"against different data"
                    )
            if "session_id" in row and row["session_id"] != record.session_id:
                raise ValueError(
                    f"eval manifest {wid}: session_id is "
                    f"{row['session_id']!r} but the split says "
                    f"{record.session_id!r}"
                )
            indices.append(idx)
        return indices

    n_active = int(selection.num_active_speakers)
    eligible = [
        i
        for i, r in enumerate(records)
        if r.num_active_speakers == n_active
        and _in_duration_band(
            r.duration,
            selection.get("min_duration"),
            selection.get("max_duration"),
        )
    ]
    cap = selection.get("per_session_cap")
    if cap is not None:
        # Balance sessions (AMI: no meeting dominates a stratum) BEFORE the
        # global cap; seeded per session so a slice re-draw is stable.
        by_session: dict[str, list[int]] = {}
        for i in eligible:
            by_session.setdefault(records[i].session_id, []).append(i)
        seed = int(selection.get("seed", 0))
        kept: list[int] = []
        for sid, idxs in by_session.items():
            if len(idxs) > int(cap):
                idxs = random.Random(f"{seed}:cap:{sid}").sample(idxs, int(cap))
            kept.extend(idxs)
        eligible = sorted(kept)
    num_windows = selection.get("num_windows")
    if num_windows is not None and len(eligible) > int(num_windows):
        rng = random.Random(int(selection.get("seed", 0)))
        eligible = sorted(rng.sample(eligible, int(num_windows)))
    return eligible


def _resolve_pinned_turns(pool_turns, prompts, record) -> list[Any]:
    """Turn a manifest row's pinned prompt spans into this session's turns.

    Every channel must be named exactly once - the project rule that every
    speaker carries a voice reference is not negotiable by manifest - every
    span must match exactly one pool turn, and no span may touch the
    evaluated window, so target leakage stays impossible even when a
    manifest asks for it.
    """
    by_channel: dict[int, Any] = {}
    for entry in prompts:
        ch = int(entry["channel"])
        if ch in by_channel:
            raise ValueError(f"{record.window_id}: channel {ch} pinned twice")
        by_channel[ch] = entry
    # Pinned channels are SOURCE channels; the result is in ROW order.
    rows = record.row_channels
    missing = [c for c in rows if c not in by_channel]
    if missing:
        raise ValueError(
            f"{record.window_id}: no pinned prompt for channel(s) {missing} "
            "- every channel needs a voice reference"
        )
    extra = sorted(c for c in by_channel if c not in rows)
    if extra:
        raise ValueError(
            f"{record.window_id}: pinned prompt for channel(s) {extra}, but "
            f"the window uses source channels {rows}"
        )

    selected: list[Any] = []
    for ch in rows:
        entry = by_channel[ch]
        start, end = float(entry["start"]), float(entry["end"])
        if _overlaps(start, end, record.t0, record.t1):
            raise ValueError(
                f"{record.window_id}: pinned prompt for channel {ch} "
                f"({start}-{end}) overlaps the evaluated window "
                f"({record.t0}-{record.t1}) - target leakage"
            )
        matches = [
            t
            for t in pool_turns
            if int(t.channel) == ch
            and spans_match(t.start, start)
            and spans_match(t.end, end)
        ]
        if not matches:
            raise ValueError(
                f"{record.window_id}: no pool turn on channel {ch} at "
                f"{start}-{end} - the corpus has moved under this manifest"
            )
        if len(matches) > 1:
            raise ValueError(
                f"{record.window_id}: pinned span {start}-{end} on channel "
                f"{ch} matches {len(matches)} pool turns"
            )
        selected.append(matches[0])
    return selected


def mask_to_turns(
    speech: torch.Tensor, turns, t0: float, fs: int, guard_sec: float
) -> torch.Tensor:
    """Zero every sample of row ``r`` outside all of row ``r``'s turn spans
    (widened by ``guard_sec``).  Used for the gt/resynth anchors on corpora
    with headset crosstalk (AMI): the annotation defines the ground truth,
    bleed from other participants is removed, and one pipeline then scores
    masked GT and generated audio alike (design note "Beyond Two Speakers",
    section 4).  ``turns`` are row-space with absolute times."""
    mask = torch.zeros_like(speech, dtype=torch.bool)
    total = speech.shape[1]
    for t in turns:
        a = max(0, int(round((t.start - t0 - guard_sec) * fs)))
        b = min(total, int(round((t.end - t0 + guard_sec) * fs)))
        if b > a:
            mask[t.channel, a:b] = True
    return speech * mask


def _reference_texts(turns, num_channels: int) -> list[str]:
    """Per-channel reference text: all of that channel's turn texts,
    space-joined in window order (the whole window is generated now)."""
    per_channel: list[list[str]] = [[] for _ in range(num_channels)]
    for turn in turns:
        per_channel[turn.channel].append(turn.text)
    return [" ".join(parts) for parts in per_channel]


def _turn_spans(turns, t0: float) -> list[dict[str, Any]]:
    """Ground-truth turn spans shifted to window time (relative to ``t0``)."""
    return [
        {
            "channel": int(turn.channel),
            "text": turn.text,
            "start": round(turn.start - t0, 6),
            "end": round(turn.end - t0, 6),
        }
        for turn in turns
    ]


def _prompt_turn_meta(turns) -> list[dict[str, Any]]:
    """Meta entries for the concatenated prompt turns, session-absolute spans,
    in concatenation (channel-ascending) order."""
    return [
        {
            "channel": int(t.channel),
            "text": t.text,
            "start": round(t.start, 6),
            "end": round(t.end, 6),
            "duration_sec": round(t.end - t.start, 6),
        }
        for t in turns
    ]


def _layout_turn_meta(turns) -> list[dict[str, Any]]:
    """Mode T layout entries: the sequence timeline the text was written
    against (seconds from conditioning-speech start, prompt included), in
    layout order.  No text: the transcripts are already in ``prompt.turns``
    and ``turns``; this block is purely the timing the model was asked for."""
    return [
        {
            "channel": int(t.channel),
            "start": round(t.start, 6),
            "end": round(t.end, 6),
        }
        for t in turns
    ]


def run_inference(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
) -> dict[str, Any]:
    """Execute the infer stage; return ``{"n_selected", "n_skipped",
    "n_timestamp_degraded"}`` (the last is always 0 outside
    ``text_format: timestamps``).

    ``training_config`` (the model / vocab / feats source) defaults to loading
    ``inference_config.training_config`` from disk, mirroring
    ``generate_dev.py``.  ``model`` / ``vocoder`` are injection seams: when
    omitted they are built lazily only for the modes that need them (``gt``
    needs neither), so tests run CPU-only with a fake vocoder.
    """
    cfg = inference_config
    mode = cfg.mode
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
    # Checked before any I/O: a typo'd or misplaced knob must fail on the
    # config, not after a manifest load.  `or "order"` also covers an
    # explicit YAML null.
    text_format = str(cfg.get("text_format", "order") or "order")
    if text_format not in _TEXT_FORMATS:
        raise ValueError(
            f"text_format must be one of {_TEXT_FORMATS}, got {text_format!r}"
        )
    if text_format == "timestamps" and mode != "generate":
        raise ValueError(
            f"text_format: timestamps requires mode: generate, got mode {mode!r} "
            "(gt/resynth never run the text preprocessor)"
        )
    # Duration policy (default ground_truth = the window length, bit-identical
    # to every run before the knob existed).  `predicted` generates the F5
    # rule's length from the prompt rates instead - generate mode, order
    # text only (a Mode T layout is anchored to the real window timeline).
    dur_cfg = cfg.get("duration", {}) or {}
    duration_source = str(dur_cfg.get("source", "ground_truth") or "ground_truth")
    if duration_source not in _DURATION_SOURCES:
        raise ValueError(
            f"duration.source must be one of {_DURATION_SOURCES}, got {duration_source!r}"
        )
    duration_scale = float(dur_cfg.get("scale", DEFAULT_DURATION_SCALE))
    duration_speed = float(dur_cfg.get("speed", 1.0))
    rate_prior_chars = dur_cfg.get("rate_prior_chars")
    if duration_source == "predicted" and (mode != "generate" or text_format != "order"):
        raise ValueError(
            "duration.source: predicted requires mode: generate and text_format: "
            f"order (got mode {mode!r}, text_format {text_format!r})"
        )

    if training_config is None:
        train_path = Path(cfg.training_config)
        if not train_path.is_absolute():
            train_path = Path(cfg.get("recipe_dir", ".")) / train_path
        training_config = OmegaConf.load(train_path)

    device = torch.device(cfg.get("device", "cpu"))
    fs = int(training_config.sample_rate)
    hop = int(training_config.hop_length)
    if text_format == "timestamps" and fs / hop != FRAMES_PER_SECOND:
        # The preprocessor's Mode T text grid is hardwired to
        # FRAMES_PER_SECOND, so a recipe whose fs/hop disagrees would
        # silently desync the text stream from the audio it describes.
        raise ValueError(
            f"text_format: timestamps needs a frame rate of {FRAMES_PER_SECOND} "
            "Hz, but this training config's sample_rate/hop_length gives "
            f"{fs / hop}"
        )

    dataset = build_dataset(
        training_config,
        cfg.dataset.split,
        inference=True,
        manifest_path=cfg.dataset.get("manifest_path"),
        dataset_root=cfg.dataset.get("dataset_root"),
    )
    pools = _build_turn_pools(dataset.records)
    manifest_path = cfg.selection.get("manifest")
    pinned_rows = None
    pinned_prompts: dict[str, Any] = {}
    if manifest_path:
        eval_header, pinned_rows = load_eval_manifest(manifest_path)
        pinned_prompts = {r["window_id"]: r["prompts"] for r in pinned_rows}
        logger.info(
            "infer selection: FROZEN eval manifest %s (%d windows, split=%s, "
            "source=%s md5=%s) - selection/prompt seeds are inert",
            manifest_path,
            len(pinned_rows),
            eval_header.get("split"),
            eval_header.get("source_manifest"),
            eval_header.get("source_manifest_md5"),
        )
    indices = _select_indices(dataset.records, cfg.selection, pinned_rows)
    logger.info(
        "infer selection: %d/%d windows (split=%s, mode=%s, seed=%s)",
        len(indices),
        len(dataset.records),
        cfg.dataset.split,
        mode,
        cfg.selection.get("seed", 0),
    )

    # Build only what this mode needs; keep gt free of model/vocoder deps.
    needs_model = mode in ("generate", "resynth")
    if needs_model and model is None:
        ckpt = cfg.get("ckpt")
        model = load_model(
            training_config,
            Path(ckpt) if ckpt else None,
            use_ema=bool(cfg.get("use_ema", True)),
            device=device,
        )
    if needs_model and vocoder is None:
        vocoder = load_vocoder(device)
    preprocessor = build_preprocessor(training_config) if mode == "generate" else None

    test_dir = Path(cfg.inference_dir) / cfg.test_name
    for sub in ("meta", "wav", "prompt", "gt", "mix"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    meta_lines: list[str] = []
    wav_lines: list[str] = []
    prompt_lines: list[str] = []
    text_lines: list[str] = []
    mix_lines: list[str] = []
    n_skipped = 0
    n_timestamp_degraded = 0  # per WINDOW, not per channel; 0 in Mode O

    prompt_cfg = cfg.prompt
    turn_min = float(prompt_cfg.get("turn_min_sec", 2.0))
    turn_max = float(prompt_cfg.get("turn_max_sec", 10.0))
    prompt_seed = prompt_cfg.get("seed", 0)
    solo_guard = float(prompt_cfg.get("solo_guard_sec", 0.0) or 0.0)
    excluded_by_session = (
        load_excluded_spans(prompt_cfg.exclude_spans)
        if prompt_cfg.get("exclude_spans")
        else {}
    )
    anchor_cfg = cfg.get("anchor", {}) or {}
    mask_cfg = anchor_cfg.get("mask_to_turns", {}) or {}
    mask_enabled = bool(mask_cfg.get("enabled", False))
    mask_guard = float(mask_cfg.get("guard_sec", 0.15))
    samp = cfg.sampling
    # tqdm renders a live bar on a tty; under a non-tty Slurm log it prints
    # one plain line per refresh (rate + ETA), so long infer runs are
    # observable either way.
    for idx in tqdm(indices, desc=f"infer[{mode}]", unit="window"):
        record = dataset.records[idx]
        pool_turns = pools.get(record.session_id, [])
        # Source channels behind the window's rows (identity unless the
        # record carries a `channels` subset, as AMI K-strata windows do).
        rows = record.row_channels
        excluded = excluded_by_session.get(record.session_id, frozenset())

        selected: list[Any] = []
        skip_channel: int | None = None
        if record.window_id in pinned_prompts:
            # A frozen manifest already made this choice; an unresolvable
            # pin is an error, never a fall back to the ladder.
            selected = _resolve_pinned_turns(
                pool_turns, pinned_prompts[record.window_id], record
            )
        else:
            for ch in rows:
                turn = _select_prompt_turn(
                    pool_turns,
                    ch,
                    record.t0,
                    record.t1,
                    turn_min,
                    turn_max,
                    prompt_seed,
                    record.window_id,
                    solo_guard=solo_guard,
                    excluded=excluded,
                )
                if turn is None:
                    skip_channel = ch
                    break
                selected.append(turn)
        if skip_channel is not None:
            n_skipped += 1
            logger.info(
                "skip %s: channel %d has no non-window prompt turn in the pool "
                "(target leakage only)",
                record.window_id,
                skip_channel,
            )
            continue

        sample = dataset[idx]
        n = sample["num_channels"]
        # Row-space window turns, captured before any Mode T / prompt mutation.
        window_turns = list(sample["turns"])
        window_speech = sample["speech"]  # (N, T_window), CPU
        # Masked anchor audio: what `gt` emits, what `resynth` vocodes, and
        # what EVERY mode writes as the gt/ copies (InteractionMetric reads
        # channels[ch].gt_wav of the same run as its W1 reference).  The
        # model input in `generate` stays the real, unmasked window.
        gt_write = (
            mask_to_turns(window_speech, window_turns, record.t0, fs, mask_guard)
            if mask_enabled
            else window_speech
        )

        audio_path = dataset.dataset_root / record.audio_relpath
        blocks = [
            read_audio_span(
                audio_path, record.sample_rate, t.start, t.end, fs, channels=rows
            )
            for t in selected
        ]
        # `source_selected` stays in SOURCE-channel space (provenance);
        # `selected` becomes ROW space, like everything downstream.
        source_selected = list(selected)
        selected = [
            dataclasses.replace(t, channel=row) for row, t in enumerate(selected)
        ]
        prompt_raw = torch.cat(blocks, dim=1)  # (N, P), CPU
        prompt_frames = prompt_raw.shape[1] // hop
        prompt_samples = prompt_frames * hop
        prompt_trimmed = prompt_raw[:, :prompt_samples]  # drop remainder from the end

        # The rule's estimate is recorded in every mode (so the anchor rows
        # carry predicted_over_gt); it sets the generated length only under
        # duration.source: predicted.
        predicted_sec = predict_generated_sec(
            [b.shape[1] / fs for b in blocks],
            [t.text for t in selected],
            _reference_texts(window_turns, n),
            duration_scale=duration_scale,
            speed=duration_speed,
            rate_prior_chars=rate_prior_chars,
        )
        gt_sec = window_speech.shape[1] / fs
        if duration_source == "predicted":
            gen_frames = max(1, int(round(predicted_sec * fs / hop)))
            region = torch.zeros(n, gen_frames * hop)  # masked anyway
        else:
            region = window_speech
        speech = torch.cat([prompt_trimmed, region], dim=1).to(device)
        total_frames = speech.shape[1] // hop

        wid = record.window_id
        rtf = None
        effective_text_format = "order"
        layout_meta: list[dict[str, Any]] | None = None
        if mode == "gt":
            gen_wavs = gt_write.cpu()
        elif mode == "resynth":
            gen_wavs = resynth_region(model, vocoder, gt_write.to(device))
        else:  # generate
            if text_format == "timestamps":
                layout_turns = prompt_window_layout(
                    selected,
                    [b.shape[1] for b in blocks],
                    window_turns,
                    record.t0,
                    fs=fs,
                )
                fps = fs / hop
                if timestamp_fits(layout_turns, 0.0, total_frames / fps, fps):
                    sample["turns"] = layout_turns
                    sample.update(
                        timestamp_text=True, target_t0=0.0, target_frames=total_frames
                    )
                    effective_text_format = "timestamps"
                    layout_meta = _layout_turn_meta(layout_turns)
                else:
                    logger.warning(
                        "%s: a turn does not fit its span; degrading to "
                        "order-only text",
                        wid,
                    )
                    n_timestamp_degraded += 1
                    sample["turns"] = list(selected) + list(window_turns)
            else:
                sample["turns"] = list(selected) + list(window_turns)
            sample = preprocessor(str(idx), sample)
            text = pad_branch_text(sample, device)
            gen_wavs, elapsed = generate_region(
                model,
                vocoder,
                speech,
                text,
                prompt_frames,
                total_frames,
                steps=int(samp.steps),
                cfg_strength=float(samp.cfg_strength),
                sway_sampling_coef=float(samp.sway_sampling_coef),
                seed=samp.get("seed"),
            )
            gen_seconds = gen_wavs.shape[1] / fs
            rtf = float(elapsed / gen_seconds) if gen_seconds > 0 else None

        ref_texts = _reference_texts(window_turns, n)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            gt_rel = f"gt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, gen_wavs[ch], fs)
            write_wav(test_dir / prompt_rel, blocks[ch][ch], fs)
            write_wav(test_dir / gt_rel, gt_write[ch].cpu(), fs)
            channels.append(
                {
                    "gen_wav": gen_rel,
                    "prompt_wav": prompt_rel,
                    "gt_wav": gt_rel,
                    "ref_text": ref_texts[ch],
                }
            )
            wav_lines.append(f"{wid}_ch{ch} {gen_rel}")
            prompt_lines.append(f"{wid}_ch{ch} {prompt_rel}")
            text_lines.append(f"{wid}_ch{ch} {ref_texts[ch]}")

        mix_rel = f"mix/{wid}.wav"
        write_wav(test_dir / mix_rel, gen_wavs.sum(dim=0).cpu() / n, fs)
        mix_lines.append(f"{wid} {mix_rel}")

        meta = {
            "window_id": wid,
            "session_id": record.session_id,
            "mode": mode,
            "sample_rate": fs,
            "num_channels": n,
            # Source column (headset index for AMI) behind each row.
            "source_channels": list(rows),
            # The scored region's length: the window, or the predicted length
            # when duration.source is predicted (measured on the written wav).
            "window_duration_sec": round(gen_wavs.shape[1] / fs, 6),
            "gt_duration_sec": round(record.t1 - record.t0, 6),
            "duration": duration_meta(
                duration_scale,
                duration_speed,
                predicted_sec,
                source=duration_source,
                gt_sec=gt_sec,
                rate_prior_chars=rate_prior_chars,
            ),
            "text_format": effective_text_format,
            # `layout` is the ONE meta key that intentionally breaks
            # gt/generate key parity, and only under `text_format:
            # timestamps` - gt can never carry it, since timestamps is
            # rejected outside generate.  The cross-mode invariant is
            # therefore "equal key sets whenever text_format is order", with
            # `layout` the sole Mode T addition; the two halves are pinned by
            # TestModeParity and TestTimestampGenerate respectively.
            **({"layout": {"turns": layout_meta}} if layout_meta is not None else {}),
            "rtf": rtf,
            # Present in every mode (key parity): whether the gt/ copies (and
            # the gt/resynth outputs) are masked to the annotated turns.
            "anchor": {"masked": bool(mask_enabled), "guard_sec": mask_guard},
            "mix_wav": mix_rel,
            "prompt": {
                "total_sec": round(prompt_samples / fs, 6),
                "total_frames": prompt_frames,
                "turns": [
                    {**m, "source_channel": int(src.channel)}
                    for m, src in zip(_prompt_turn_meta(selected), source_selected)
                ],
            },
            "channels": channels,
            "turns": _turn_spans(window_turns, record.t0),
        }
        meta_rel = f"meta/{wid}.json"
        (test_dir / meta_rel).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_lines.append(f"{wid} {meta_rel}")

    _write_scp(test_dir / "meta.scp", meta_lines)
    _write_scp(test_dir / "wav.scp", wav_lines)
    _write_scp(test_dir / "prompt.scp", prompt_lines)
    _write_scp(test_dir / "text.scp", text_lines)
    _write_scp(test_dir / "mix.scp", mix_lines)

    n_selected = len(meta_lines)
    logger.info(
        "infer done: %d generated, %d skipped, %d timestamp-degraded -> %s",
        n_selected,
        n_skipped,
        n_timestamp_degraded,
        test_dir,
    )
    return {
        "n_selected": n_selected,
        "n_skipped": n_skipped,
        "n_timestamp_degraded": n_timestamp_degraded,
    }


def _write_scp(path: Path, lines: Sequence[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
