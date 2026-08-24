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
from egs3.conversational.tts.src.timestamp_layout import prompt_window_layout

logger = logging.getLogger(__name__)

_EPS = 1e-6
_MODES = ("generate", "gt", "resynth")
_TEXT_FORMATS = ("order", "timestamps")


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


def _select_prompt_turn(
    pool_turns: Sequence[Any],
    channel: int,
    t0: float,
    t1: float,
    turn_min: float,
    turn_max: float,
    seed: Any,
    window_id: str,
):
    """Pick one prompt turn for ``channel`` via the relaxation ladder (module
    docstring).  Returns ``None`` iff even the loosest tier (non-window) is
    empty - target leakage is never allowed, so that tier is never relaxed.
    """
    non_window = [
        t
        for t in pool_turns
        if t.channel == channel and not _overlaps(t.start, t.end, t0, t1)
    ]
    if not non_window:
        return None

    def _is_solo(turn) -> bool:
        return not any(
            _overlaps(turn.start, turn.end, other.start, other.end)
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


def _select_indices(records, selection) -> list[int]:
    """Filtered, seeded, capped window indices (sorted for determinism)."""
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
    num_windows = selection.get("num_windows")
    if num_windows is not None and len(eligible) > int(num_windows):
        rng = random.Random(int(selection.get("seed", 0)))
        eligible = sorted(rng.sample(eligible, int(num_windows)))
    return eligible


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
    indices = _select_indices(dataset.records, cfg.selection)
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
    samp = cfg.sampling
    # tqdm renders a live bar on a tty; under a non-tty Slurm log it prints
    # one plain line per refresh (rate + ETA), so long infer runs are
    # observable either way.
    for idx in tqdm(indices, desc=f"infer[{mode}]", unit="window"):
        record = dataset.records[idx]
        pool_turns = pools.get(record.session_id, [])

        selected: list[Any] = []
        skip_channel: int | None = None
        for ch in range(record.num_channels):
            turn = _select_prompt_turn(
                pool_turns,
                ch,
                record.t0,
                record.t1,
                turn_min,
                turn_max,
                prompt_seed,
                record.window_id,
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
        window_speech = sample["speech"]  # (N, T_window), CPU

        audio_path = dataset.dataset_root / record.audio_relpath
        blocks = [
            read_audio_span(audio_path, record.sample_rate, t.start, t.end, fs)
            for t in selected
        ]
        prompt_raw = torch.cat(blocks, dim=1)  # (N, P), CPU
        prompt_frames = prompt_raw.shape[1] // hop
        prompt_samples = prompt_frames * hop
        prompt_trimmed = prompt_raw[:, :prompt_samples]  # drop remainder from the end

        speech = torch.cat([prompt_trimmed, window_speech], dim=1).to(device)
        total_frames = speech.shape[1] // hop

        wid = record.window_id
        rtf = None
        effective_text_format = "order"
        layout_meta: list[dict[str, Any]] | None = None
        if mode == "gt":
            gen_wavs = window_speech.cpu()
        elif mode == "resynth":
            gen_wavs = resynth_region(model, vocoder, window_speech.to(device))
        else:  # generate
            if text_format == "timestamps":
                layout_turns = prompt_window_layout(
                    selected,
                    [b.shape[1] for b in blocks],
                    sample["turns"],
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
                    sample["turns"] = list(selected) + list(sample["turns"])
            else:
                sample["turns"] = list(selected) + list(sample["turns"])
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

        ref_texts = _reference_texts(record.turns, n)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            gt_rel = f"gt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, gen_wavs[ch], fs)
            write_wav(test_dir / prompt_rel, blocks[ch][ch], fs)
            write_wav(test_dir / gt_rel, window_speech[ch].cpu(), fs)
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
            "window_duration_sec": round(record.t1 - record.t0, 6),
            "text_format": effective_text_format,
            **({"layout": {"turns": layout_meta}} if layout_meta is not None else {}),
            "rtf": rtf,
            "mix_wav": mix_rel,
            "prompt": {
                "total_sec": round(prompt_samples / fs, 6),
                "total_frames": prompt_frames,
                "turns": _prompt_turn_meta(selected),
            },
            "channels": channels,
            "turns": _turn_spans(record.turns, record.t0),
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
