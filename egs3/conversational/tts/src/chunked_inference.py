"""Chunked ``infer`` stage for the external (CoVoMix2) test set.

Splits each dialogue's turn list at turn boundaries into chunks and runs one
ODE call per chunk: call k is conditioned on the FULL generated audio of
chunk k-1 (both channels) and the text of chunks k-1 and k, so every call
stays inside the short-generation regime where the model is grounded (the
full-run WER curve doubles per ~15 s of generated length).  Chunk k-1's
generated audio corresponds to known text BY CONSTRUCTION - chunks are whole
turns - so no aligner is ever needed.

Runs alongside ``src/external_inference.py`` exactly as that module runs
alongside the SSSD path: nothing here is imported by the other infer modes,
so ``generate_external`` and the SSSD modes stay bit-reproducible.

Chunk policy (config ``chunk:``, exactly one key):
* ``turns: N`` - every chunk is N consecutive turns (N=2 guarantees both
  speakers appear in every chunk given the test set's strict alternation).
* ``target_sec: S`` - pack turns until the chunk's PREDICTED duration would
  exceed S; a single turn longer than S still becomes its own oversized
  chunk (flagged in meta and counted in the log), never split inside.

``chunk.cover_all_speakers: true`` (``target_sec`` only, rejected otherwise)
additionally holds every non-final chunk open until it has heard from every
channel, so conditioning chunk k+1 never loses a speaker's voice reference -
a chunk held open this way may exceed the target and is flagged oversized
like any other.

Seams: ``chunk.cross_fade_sec`` blends each join with an equal-power
cross-fade (:func:`crossfade_concat`); 0.0 (the default) keeps the original
hard concat bit-for-bit.  The fade is assembly-only - conditioning always
uses chunk k-1's RAW generated audio, so the sampled audio itself is
identical at every fade setting and only the written wavs differ.  Each seam
shortens the output by the fade length, so ``window_duration_sec`` (measured
on the written wav) is ``(n_chunks - 1) * cross_fade_sec`` short of the
predicted duration.  Chunk k still starts strictly after chunk k-1's turns,
so cross-seam turn overlap stays structurally impossible - watch
``overlap_per_min``.

Conditioning hygiene (the mirror image of the cross-fade contract: it
transforms what call k SEES, never what is written): measured on the
CoVoMix2 eval, conditioning each chunk on raw generated audio compounds two
biases - active-frame RMS grows ~+50% per hop (flat in the unchunked
control) and non-speech "junk" on idle channels is inherited and never
recovers (escalation 0.83 vs 0.35, junk fraction 0.04 -> 0.50 by chunk 5).
Two independent ``chunk`` knobs break the two feedback loops, both default
OFF so existing configs reproduce bit-for-bit:

* ``cond_silence_gate: true`` - replace every conditioning sample outside
  Silero-detected speech regions (:func:`silence_gate`), so an idle
  channel stops seeding the junk feedback loop.  ``cond_gate_threshold``
  (default 0.15, the eval metric's setting - low on purpose: keep
  degraded-but-real speech, cut only confident junk) and
  ``cond_gate_fill`` (what replaces non-speech: ``room_tone``, the
  default - the channel's own prompt's noise floor - or ``zeros``, the v1
  behavior that measurably cost ~0.9 pt WER because training idle
  channels are never digitally silent) both require the gate to be on.
* ``cond_loudness_norm: true`` - rescale each conditioning channel to the
  active-frame RMS of that channel's REAL prompt (:func:`match_active_rms`,
  anchor computed once at round 0), so gain drift cannot compound however
  long the chain runs.

Gate runs before normalization so junk cannot skew the RMS estimate.  Per
round the applied ``gains`` / ``gated_frac`` are recorded under the chunk's
``conditioning`` meta key (absent at round 0 and when both knobs are off).

The same OOD-silence problem exists at round 0: ``_prompt_blocks`` fills
the off rows of each prompt block with digital zeros.  The top-level
``prompt_fill`` config key (``zeros``, the default - bit-identical - or
``room_tone``) selects the fill there; it is recorded in the meta under
``prompt.fill``.

DETERMINISM CONTRACT - a documented departure from ``generate_external``:
sharding is BY DIALOGUE (a chunk chain cannot cross shards), and each shard
plans its round batches over its own dialogues, so outputs are a pure
function of the full config INCLUDING ``shard_count`` - rerunning any shard
reproduces it bit-for-bit, but changing ``shard_count`` redraws equally-valid
samples (exactly like changing the batching knobs does on the unchunked
path).  Fix ``shard_count`` in eval configs like any other sampling knob.
Round k reseeds with ``sampling.seed + k`` so consecutive chunks of the same
batch shape never reuse an identical noise draw.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.src.external_inference import (
    ACTIVE_RMS_FRAME_SEC,
    ACTIVE_RMS_THRESHOLD,
    PROMPT_FILLS,
    _load_prompt_wav,
    _probe_duration_sec,
    _prompt_blocks,
    _prompt_turns,
    room_tone,
    tile_to,
)
from egs3.conversational.tts.src.external_testset import (
    DEFAULT_DURATION_SCALE,
    ExternalRecord,
    assign_shard,
    duration_meta,
    load_covomix2_testset,
    plan_batches,
    select_records,
)
from egs3.conversational.tts.src.generation import (
    GenerationItem,
    build_preprocessor,
    generate_batch,
    load_model,
    load_vocoder,
    pad_branch_text,
    write_wav,
)

# Output-format helpers, reused verbatim so the infer paths can never drift
# apart in what they write (same import external_inference.py does).
from egs3.conversational.tts.src.inference import _reference_texts, _write_scp

logger = logging.getLogger(__name__)

MODE = "generate_external_chunked"


def estimate_turn_secs(
    record: ExternalRecord,
    prompt_seconds: Sequence[float],
    *,
    duration_scale: float,
    speed: float,
) -> list[float]:
    """Per-turn predicted seconds under the F5 per-speaker rate rule.

    The same rule as :func:`external_testset.estimate_duration_sec`, split
    per turn instead of summed: each turn is priced at its own speaker's
    prompt-measured seconds-per-character.  ``sum(estimate_turn_secs(...))``
    equals ``estimate_duration_sec(...)`` by construction, so chunking never
    changes the duration policy - only where the ODE calls are cut.
    """
    if len(prompt_seconds) != record.num_channels:
        raise ValueError(
            f"{record.dialogue_id}: got {len(prompt_seconds)} prompt durations "
            f"for {record.num_channels} channels"
        )
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    rates = []
    for ch, (prompt_sec, prompt) in enumerate(zip(prompt_seconds, record.prompts)):
        prompt_chars = len(prompt.text.encode("utf-8"))
        if prompt_sec <= 0 or prompt_chars <= 0:
            raise ValueError(
                f"{record.dialogue_id}: channel {ch} prompt is degenerate "
                f"({prompt_sec:.3f}s, {prompt_chars} chars)"
            )
        rates.append(prompt_sec / prompt_chars)
    return [
        len(turn.text.encode("utf-8"))
        * rates[turn.channel]
        * float(duration_scale)
        / float(speed)
        for turn in record.turns
    ]


def split_turns(
    turn_secs: Sequence[float],
    *,
    turns: int | None = None,
    target_sec: float | None = None,
    channels: Sequence[int] | None = None,
    num_channels: int | None = None,
    cover_all_speakers: bool = False,
) -> list[tuple[int, int]]:
    """Split turn indices into chunks; returns half-open ``(start, end)``
    ranges that partition ``range(len(turn_secs))`` in order.

    Exactly one policy must be given.  ``turns``: fixed count per chunk (the
    last chunk keeps the remainder).  ``target_sec``: greedy packing - a turn
    joins the current chunk unless that would push the chunk's predicted
    total past the target; a single turn longer than the target still gets
    its own chunk (the policy bounds growth, it never splits inside a turn).

    ``cover_all_speakers`` (``target_sec`` only) additionally forbids a chunk
    from closing until it contains at least one turn from EVERY channel in
    ``range(num_channels)`` - conditioning chunk k on chunk k-1 loses a
    speaker's voice reference whenever k-1 never lets that speaker talk, so
    coverage is a conditioning guarantee, not a packing nicety.  A chunk held
    open for coverage may exceed the target (flagged oversized by the
    caller).  The FINAL chunk is exempt: it closes at the end of the turn
    list and never conditions anything.
    """
    if (turns is None) == (target_sec is None):
        raise ValueError("exactly one of `turns` / `target_sec` must be set")
    if not len(turn_secs):
        raise ValueError("no turns to split")
    if cover_all_speakers:
        if target_sec is None:
            raise ValueError("cover_all_speakers requires the target_sec policy")
        if channels is None or num_channels is None:
            raise ValueError(
                "cover_all_speakers needs `channels` and `num_channels`"
            )
        if len(channels) != len(turn_secs):
            raise ValueError(
                f"got {len(channels)} channels for {len(turn_secs)} turns"
            )
    if turns is not None:
        n = int(turns)
        if n < 1:
            raise ValueError(f"turns must be >= 1, got {turns}")
        return [(i, min(i + n, len(turn_secs))) for i in range(0, len(turn_secs), n)]
    target = float(target_sec)
    if target <= 0:
        raise ValueError(f"target_sec must be > 0, got {target_sec}")
    required = set(range(int(num_channels))) if cover_all_speakers else None
    ranges: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    seen: set[int] = set()
    for i, sec in enumerate(turn_secs):
        if (
            i > start
            and acc + float(sec) > target
            and (required is None or required <= seen)
        ):
            ranges.append((start, i))
            start, acc, seen = i, 0.0, set()
        acc += float(sec)
        if required is not None:
            seen.add(int(channels[i]))
    ranges.append((start, len(turn_secs)))
    return ranges


def crossfade_concat(
    chunk_wavs: Sequence[torch.Tensor], fade_samples: int
) -> torch.Tensor:
    """Concat ``(channels, samples)`` chunks with an equal-power cross-fade.

    At every seam the last ``fade_samples`` of the left chunk overlap the
    first ``fade_samples`` of the right one under cos/sin gains sampled at
    window midpoints, so per-sample ``g_down**2 + g_up**2 == 1`` exactly:
    adjacent chunks come from different noise draws and are therefore
    uncorrelated, and uncorrelated signals add in POWER - linear gains would
    dip the seam loudness by 3 dB at its midpoint.  The fade clamps to the
    shorter neighbour, ``fade_samples=0`` reduces to ``torch.cat`` (the hard
    concat), and every sample outside a seam is bit-identical to its chunk.
    """
    if fade_samples < 0:
        raise ValueError(f"fade_samples must be >= 0, got {fade_samples}")
    out = chunk_wavs[0]
    for nxt in chunk_wavs[1:]:
        f = min(int(fade_samples), out.shape[1], nxt.shape[1])
        if f == 0:
            out = torch.cat([out, nxt], dim=1)
            continue
        t = (torch.arange(f, dtype=out.dtype, device=out.device) + 0.5) / f
        g_down = torch.cos(t * torch.pi / 2)
        g_up = torch.sin(t * torch.pi / 2)
        seam = out[:, out.shape[1] - f :] * g_down + nxt[:, :f] * g_up
        out = torch.cat([out[:, : out.shape[1] - f], seam, nxt[:, f:]], dim=1)
    return out


# ACTIVE_RMS_* live in external_inference (shared with room_tone).
DEFAULT_GATE_THRESHOLD = 0.15
DEFAULT_GATE_FILL = "room_tone"
COND_GAIN_CLAMP = (0.1, 10.0)


def active_rms(wav: torch.Tensor, fs: int) -> float | None:
    """Mean RMS over the ACTIVE 20 ms frames of a 1-D wave, or ``None``.

    Plain RMS would be diluted by however much of the signal is silence, so
    a mostly-idle channel would demand a huge gain to "match" a continuous
    prompt; framewise gating (RMS > 1e-3, same floor the eval analysis
    used) prices only the speech itself.  ``None`` means no active frame -
    the caller must not derive a gain from nothing.
    """
    frame = int(fs * ACTIVE_RMS_FRAME_SEC)
    n = wav.shape[0] // frame
    if n == 0:
        return None
    rms = wav[: n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt()
    active = rms > ACTIVE_RMS_THRESHOLD
    if not bool(active.any()):
        return None
    return float(rms[active].mean())


def _silero_speech_regions(
    wav: torch.Tensor, fs: int, threshold: float
) -> list[tuple[int, int]]:
    """Speech spans of a 1-D wave as ``(start, end)`` sample ranges at
    ``fs``, from the Silero VAD bundled with faster-whisper (resampled to
    its native 16 kHz and mapped back)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "chunk.cond_silence_gate needs the Silero VAD bundled with "
            "faster-whisper; install faster-whisper or inject a "
            "speech_regions_fn"
        ) from exc
    import torchaudio.functional as AF

    wav16 = AF.resample(wav.detach().cpu().float(), fs, 16000).numpy()
    spans = get_speech_timestamps(wav16, vad_options=VadOptions(threshold=threshold))
    scale = fs / 16000.0
    return [(int(s["start"] * scale), int(s["end"] * scale)) for s in spans]


def silence_gate(
    wav: torch.Tensor,
    fs: int,
    *,
    threshold: float,
    speech_regions_fn=None,
    fill: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Replace every sample outside detected speech regions, per channel.

    ``wav`` is ``(channels, samples)``; returns ``(gated, gated_frac)``
    with the input untouched.  ``speech_regions_fn(ch_wav, fs, threshold)``
    defaults to Silero via faster-whisper; tests inject a fake.  Non-speech
    is replaced by ``fill[ch]`` tiled across the channel (the in-domain
    room tone of that channel's real prompt), or by zeros when ``fill`` is
    None - v1 behavior, kept because it measurably COSTS WER: training
    idle channels are never digitally silent, so zeros are out-of-domain
    conditioning.  Either way an idle channel stops seeding the junk
    feedback loop.
    """
    fn = speech_regions_fn or _silero_speech_regions
    total = wav.shape[1]
    gated = torch.zeros_like(wav)
    replaced = []
    for ch in range(wav.shape[0]):
        mask = torch.zeros(total, dtype=torch.bool, device=wav.device)
        for a, b in fn(wav[ch], fs, threshold):
            mask[max(0, int(a)) : min(total, int(b))] = True
        if fill is not None:
            base = tile_to(fill[ch], total).to(dtype=wav.dtype, device=wav.device)
        else:
            base = torch.zeros((), dtype=wav.dtype, device=wav.device)
        gated[ch] = torch.where(mask, wav[ch], base)
        replaced.append(1.0 - float(mask.float().mean()) if total else 0.0)
    return gated, replaced


def match_active_rms(
    wav: torch.Tensor,
    targets: Sequence[float | None],
    fs: int,
    *,
    clamp: tuple[float, float] = COND_GAIN_CLAMP,
) -> tuple[torch.Tensor, list[float]]:
    """Scale each channel so its active-frame RMS matches ``targets[ch]``.

    A ``None`` target or an all-silent channel passes through at gain 1.0;
    gains clamp to ``clamp`` so a degenerate estimate can never blast or
    kill a channel.  Returns ``(scaled, gains)`` without touching the
    input.
    """
    out = wav.clone()
    gains = []
    for ch in range(wav.shape[0]):
        target = targets[ch]
        current = active_rms(wav[ch], fs)
        if target is None or current is None:
            gains.append(1.0)
            continue
        gain = min(max(target / current, clamp[0]), clamp[1])
        out[ch] = wav[ch] * gain
        gains.append(float(gain))
    return out, gains


@dataclass(frozen=True)
class CondHygiene:
    """Conditioning-hygiene knobs from the ``chunk`` block (defaults off)."""

    silence_gate: bool = False
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    gate_fill: str = DEFAULT_GATE_FILL
    loudness_norm: bool = False

    @property
    def enabled(self) -> bool:
        return self.silence_gate or self.loudness_norm


@dataclass(frozen=True)
class CondComposition:
    """What round r > 0 conditions on: ``[prompt?] + [last H generated chunks]``.

    ``history_chunks`` counts previous generated chunks (most recent last);
    ``0`` is none and ``-1`` is all of them.  The default ``(False, 1)`` is
    today's previous-chunk-only conditioning, bit-for-bit.
    """

    include_prompt: bool = False
    history_chunks: int = 1


def _apply_cond_hygiene(
    wav: torch.Tensor,
    *,
    fs: int,
    cond: CondHygiene,
    targets: Sequence[float | None] | None,
    fill: Sequence[torch.Tensor] | None,
    speech_regions_fn,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Gate first (junk must not skew the RMS estimate), then normalize."""
    info: dict[str, list[float]] = {}
    if cond.silence_gate:
        wav, replaced = silence_gate(
            wav,
            fs,
            threshold=cond.gate_threshold,
            speech_regions_fn=speech_regions_fn,
            fill=fill,
        )
        info["gated_frac"] = [round(f, 6) for f in replaced]
    if cond.loudness_norm:
        if targets is None:
            targets = [None] * wav.shape[0]
        wav, gains = match_active_rms(wav, targets, fs)
        info["gains"] = [round(g, 6) for g in gains]
    return wav, info


def call_turns(
    record: ExternalRecord, ranges: Sequence[tuple[int, int]], k: int
) -> list[Turn]:
    """Turns conditioning ODE call ``k``.

    Call 0 is today's external layout: the per-channel prompt turns followed
    by chunk 0's turns.  Call k > 0 covers exactly the audio span of the
    call - chunk k-1 (the continuation prompt) then chunk k - which keeps
    the ``<turn>``/``<OTHER>`` budget consistent with the conditioning audio
    by construction.  Ranges are contiguous, so this is one slice.
    """
    if k == 0:
        a, b = ranges[0]
        return _prompt_turns(record) + list(record.turns[a:b])
    prev_start = ranges[k - 1][0]
    end = ranges[k][1]
    return list(record.turns[prev_start:end])


@dataclass
class _ChunkPlan:
    """One dialogue's chunk chain, fixed before any generation happens."""

    ranges: list[tuple[int, int]]
    chunk_secs: list[float]
    gen_frames: list[int]
    oversized: list[bool]

    @property
    def n_chunks(self) -> int:
        return len(self.ranges)


def _plan_dialogue(
    record: ExternalRecord,
    prompt_secs: Sequence[float],
    *,
    chunk_cfg: dict[str, Any],
    duration_scale: float,
    speed: float,
    fs: int,
    hop: int,
    cover_all_speakers: bool = False,
) -> _ChunkPlan:
    turn_secs = estimate_turn_secs(
        record, prompt_secs, duration_scale=duration_scale, speed=speed
    )
    target = chunk_cfg.get("target_sec")
    ranges = split_turns(
        turn_secs,
        turns=chunk_cfg.get("turns"),
        target_sec=target,
        channels=[t.channel for t in record.turns],
        num_channels=record.num_channels,
        cover_all_speakers=cover_all_speakers,
    )
    chunk_secs = [sum(turn_secs[a:b]) for a, b in ranges]
    return _ChunkPlan(
        ranges=ranges,
        chunk_secs=chunk_secs,
        gen_frames=[max(1, round(sec * fs / hop)) for sec in chunk_secs],
        oversized=[target is not None and sec > float(target) for sec in chunk_secs],
    )


def _validated_chunk_cfg(
    cfg,
) -> tuple[dict[str, Any], float, CondHygiene, CondComposition, bool]:
    """Return ``(policy, cross_fade_sec, cond, comp, cover_all_speakers)`` from the
    ``chunk`` block.

    The policy stays exactly one of ``turns`` / ``target_sec``;
    ``cross_fade_sec`` is an independent seam knob, ``comp`` is a
    conditioning-composition knob, ``cover_all_speakers`` is an independent
    coverage knob restricted to the ``target_sec`` policy, and the
    conditioning-hygiene knobs (``cond_silence_gate``, ``cond_gate_threshold``,
    ``cond_loudness_norm``) are independent conditioning knobs - all default off,
    so existing configs reproduce bit-for-bit.
    """
    raw = cfg.get("chunk")
    if raw is None:
        raise ValueError(
            f"mode {MODE!r} requires a `chunk` policy block "
            "({turns: N} or {target_sec: S})"
        )
    chunk_cfg = OmegaConf.to_container(raw, resolve=True)
    unknown = set(chunk_cfg) - {
        "turns",
        "target_sec",
        "cross_fade_sec",
        "cond_silence_gate",
        "cond_gate_threshold",
        "cond_gate_fill",
        "cond_loudness_norm",
        "cond_include_prompt",
        "cond_history_chunks",
        "cover_all_speakers",
    }
    if unknown:
        raise ValueError(f"unknown chunk keys: {sorted(unknown)}")
    cover = bool(chunk_cfg.pop("cover_all_speakers", False) or False)
    cross_fade_sec = float(chunk_cfg.pop("cross_fade_sec", None) or 0.0)
    if cross_fade_sec < 0:
        raise ValueError(f"chunk.cross_fade_sec must be >= 0, got {cross_fade_sec}")
    gate_on = bool(chunk_cfg.pop("cond_silence_gate", False) or False)
    threshold_raw = chunk_cfg.pop("cond_gate_threshold", None)
    gate_fill_raw = chunk_cfg.pop("cond_gate_fill", None)
    norm_on = bool(chunk_cfg.pop("cond_loudness_norm", False) or False)
    if threshold_raw is not None and not gate_on:
        raise ValueError("chunk.cond_gate_threshold requires cond_silence_gate: true")
    if gate_fill_raw is not None and not gate_on:
        raise ValueError("chunk.cond_gate_fill requires cond_silence_gate: true")
    threshold = (
        DEFAULT_GATE_THRESHOLD if threshold_raw is None else float(threshold_raw)
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"chunk.cond_gate_threshold must be in (0, 1), got {threshold}"
        )
    gate_fill = DEFAULT_GATE_FILL if gate_fill_raw is None else str(gate_fill_raw)
    if gate_fill not in PROMPT_FILLS:
        raise ValueError(
            f"chunk.cond_gate_fill must be one of {PROMPT_FILLS}, got {gate_fill!r}"
        )
    cond = CondHygiene(
        silence_gate=gate_on,
        gate_threshold=threshold,
        gate_fill=gate_fill,
        loudness_norm=norm_on,
    )
    include_prompt = bool(chunk_cfg.pop("cond_include_prompt", False) or False)
    history_raw = chunk_cfg.pop("cond_history_chunks", None)
    history_chunks = 1 if history_raw is None else int(history_raw)
    if history_chunks < -1:
        raise ValueError(
            f"chunk.cond_history_chunks must be >= -1 (-1 = all), got {history_chunks}"
        )
    if not include_prompt and history_chunks == 0:
        raise ValueError(
            "chunk.cond_history_chunks: 0 leaves no conditioning - it requires "
            "cond_include_prompt: true"
        )
    comp = CondComposition(include_prompt=include_prompt, history_chunks=history_chunks)
    set_keys = [k for k, v in chunk_cfg.items() if v is not None]
    if len(set_keys) != 1:
        raise ValueError(
            "exactly one of chunk.turns / chunk.target_sec must be set, "
            f"got {chunk_cfg}"
        )
    policy = {k: chunk_cfg[k] for k in set_keys}
    if cover and "target_sec" not in policy:
        raise ValueError("chunk.cover_all_speakers requires the target_sec policy")
    return policy, cross_fade_sec, cond, comp, cover


def run_chunked_inference(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
    speech_regions_fn=None,
) -> dict[str, Any]:
    """Execute the chunked external infer stage; return counts.

    Same injection seams as ``run_external_inference`` so tests drive this
    CPU-only with the tiny random-init DiT and a fake vocoder;
    ``speech_regions_fn`` additionally lets tests fake the VAD behind
    ``chunk.cond_silence_gate``.
    """
    cfg = inference_config
    mode = cfg.get("mode")
    if mode != MODE:
        raise ValueError(f"expected mode {MODE!r}, got {mode!r}")
    chunk_cfg, cross_fade_sec, cond, comp, cover_all_speakers = _validated_chunk_cfg(
        cfg
    )
    prompt_fill = str(cfg.get("prompt_fill", "zeros") or "zeros")
    if prompt_fill not in PROMPT_FILLS:
        raise ValueError(
            f"prompt_fill must be one of {PROMPT_FILLS}, got {prompt_fill!r}"
        )

    if training_config is None:
        train_path = Path(cfg.training_config)
        if not train_path.is_absolute():
            train_path = Path(cfg.get("recipe_dir", ".")) / train_path
        training_config = OmegaConf.load(train_path)

    device = torch.device(cfg.get("device", "cpu"))
    fs = int(training_config.sample_rate)
    hop = int(training_config.hop_length)

    testset = cfg.testset
    records = load_covomix2_testset(
        testset.root,
        testset.librispeech_root,
        OmegaConf.to_container(training_config, resolve=True)["dataset"][
            "preprocessor"
        ]["token_list"],
        num_channels=int(testset.get("num_channels", 2)),
    )

    dur_cfg = cfg.get("duration", {})
    duration_scale = float(dur_cfg.get("scale", DEFAULT_DURATION_SCALE))
    speed = float(dur_cfg.get("speed", 1.0))

    prompt_secs = [
        [_probe_duration_sec(p.audio_path) for p in r.prompts] for r in records
    ]
    plans = [
        _plan_dialogue(
            r,
            secs,
            chunk_cfg=chunk_cfg,
            duration_scale=duration_scale,
            speed=speed,
            fs=fs,
            hop=hop,
            cover_all_speakers=cover_all_speakers,
        )
        for r, secs in zip(records, prompt_secs)
    ]
    predicted = [sum(plan.chunk_secs) for plan in plans]

    shard_count = int(cfg.selection.get("shard_count", 1) or 1)
    shard_index = int(cfg.selection.get("shard_index", 0) or 0)
    indices, exclusions = select_records(records, predicted, cfg.selection)

    # Shard BY DIALOGUE: a chunk chain cannot cross processes.  Cost is the
    # audio the chain's ODE calls actually integrate: call 0 is prompts +
    # chunk 0, call k is chunk k-1 + chunk k.
    def _chain_cost(idx: int) -> float:
        plan = plans[idx]
        cost = sum(prompt_secs[idx]) + plan.chunk_secs[0]
        for k in range(1, plan.n_chunks):
            cost += plan.chunk_secs[k - 1] + plan.chunk_secs[k]
        return cost

    selected = set(indices)
    chain_costs = [
        _chain_cost(i) if i in selected else 0.0 for i in range(len(records))
    ]
    my_indices = assign_shard(indices, chain_costs, shard_index, shard_count)

    for idx in my_indices:
        record = records[idx]
        empty = [c for c, chars in enumerate(record.channel_chars) if chars == 0]
        if empty:
            raise ValueError(
                f"{record.dialogue_id}: channel(s) {empty} have no turns at "
                f"num_channels={record.num_channels}; every selected dialogue "
                "needs at least one turn per channel - exclude it via "
                "selection.dialogue_ids"
            )

    max_rounds = max((plans[i].n_chunks for i in my_indices), default=0)
    n_oversized = sum(sum(plans[i].oversized) for i in my_indices)
    logger.info(
        "chunked external infer: %d/%d dialogues (%d out of duration band, "
        "%d not sampled, %d other shards; policy=%s, scale=%.4f, speed=%.3f) "
        "in %d rounds, %d oversized chunks",
        len(my_indices),
        len(records),
        exclusions["n_out_of_band"],
        exclusions["n_not_sampled"],
        len(indices) - len(my_indices),
        chunk_cfg,
        duration_scale,
        speed,
        max_rounds,
        n_oversized,
    )

    if model is None:
        ckpt = cfg.get("ckpt")
        model = load_model(
            training_config,
            Path(ckpt) if ckpt else None,
            use_ema=bool(cfg.get("use_ema", True)),
            device=device,
        )
    if vocoder is None:
        vocoder = load_vocoder(device)
    preprocessor = build_preprocessor(training_config)

    test_dir = Path(cfg.inference_dir) / cfg.test_name
    for sub in ("meta", "wav", "prompt", "mix"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    batching = cfg.get("batching", {}) or {}
    max_batch_audio_sec = batching.get("max_batch_audio_sec")
    max_batch_dialogues = batching.get("max_batch_dialogues")
    samp = cfg.sampling
    autocast_dtype = samp.get("autocast_dtype")
    base_seed = samp.get("seed")

    # Per-dialogue chain state, filled round by round.
    state: dict[int, dict[str, Any]] = {
        idx: {
            "blocks": None,
            "prompt_frames0": None,
            "prev_wav": None,
            "chunk_wavs": [],
            "chunk_meta": [],
        }
        for idx in my_indices
    }
    n_batches = 0

    for rnd in range(max_rounds):
        active = [idx for idx in my_indices if rnd < plans[idx].n_chunks]
        round_costs = [0.0] * len(records)
        prepared_inputs: dict[int, dict[str, Any]] = {}
        for idx in active:
            record = records[idx]
            n = record.num_channels
            cond_info: dict[str, list[float]] = {}
            if rnd == 0:
                prompt_wavs = [
                    _load_prompt_wav(p.audio_path, fs) for p in record.prompts
                ]
                blocks = _prompt_blocks(prompt_wavs, n, fill=prompt_fill, fs=fs)
                prompt_raw = torch.cat(blocks, dim=1)
                state[idx]["blocks"] = blocks
                if cond.loudness_norm:
                    # Gain anchor: each channel's REAL prompt, so the level
                    # can never drift however many hops the chain runs.
                    state[idx]["cond_targets"] = [
                        active_rms(blocks[ch][ch], fs) for ch in range(n)
                    ]
                if cond.silence_gate and cond.gate_fill == "room_tone":
                    # In-domain silence for the gate: each channel's own
                    # prompt's noise floor (zeros are out-of-domain and
                    # measurably cost WER).
                    state[idx]["cond_fill"] = [
                        room_tone(blocks[ch][ch], fs) for ch in range(n)
                    ]
            else:
                prompt_raw = state[idx]["prev_wav"]
                if cond.enabled:
                    prompt_raw, cond_info = _apply_cond_hygiene(
                        prompt_raw,
                        fs=fs,
                        cond=cond,
                        targets=state[idx].get("cond_targets"),
                        fill=state[idx].get("cond_fill"),
                        speech_regions_fn=speech_regions_fn,
                    )
            prompt_frames = prompt_raw.shape[1] // hop
            prompt_trimmed = prompt_raw[:, : prompt_frames * hop]
            if rnd == 0:
                state[idx]["prompt_frames0"] = prompt_frames
            gen_frames = plans[idx].gen_frames[rnd]
            speech = torch.cat(
                [prompt_trimmed, torch.zeros(n, gen_frames * hop)], dim=1
            ).to(device)
            sample = {
                "turns": call_turns(record, plans[idx].ranges, rnd),
                "num_channels": n,
            }
            sample = preprocessor(record.dialogue_id, sample)
            text = pad_branch_text(sample, device)
            prepared_inputs[idx] = {
                "item": GenerationItem(
                    speech=speech,
                    text=text,
                    prompt_frames=prompt_frames,
                    total_frames=prompt_frames + gen_frames,
                ),
                "gen_frames": gen_frames,
                "cond_info": cond_info,
            }
            round_costs[idx] = (prompt_frames + gen_frames) * hop / fs

        batches = plan_batches(
            active,
            round_costs,
            max_batch_audio_sec=(
                float(max_batch_audio_sec) if max_batch_audio_sec is not None else None
            ),
            max_batch_dialogues=(
                int(max_batch_dialogues) if max_batch_dialogues is not None else None
            ),
        )
        seed = base_seed if base_seed is None else int(base_seed) + rnd
        progress = tqdm(
            enumerate(batches),
            desc=f"infer[{MODE}] round {rnd + 1}/{max_rounds}",
            unit="batch",
            total=len(batches),
        )
        for batch_id, batch_indices in progress:
            items = [prepared_inputs[idx]["item"] for idx in batch_indices]
            gen_wav_list, elapsed = generate_batch(
                model,
                vocoder,
                items,
                steps=int(samp.steps),
                cfg_strength=float(samp.cfg_strength),
                sway_sampling_coef=float(samp.sway_sampling_coef),
                seed=seed,
                autocast_dtype=autocast_dtype,
            )
            n_batches += 1
            batch_gen_sec = sum(w.shape[1] for w in gen_wav_list) / fs
            batch_rtf = float(elapsed / batch_gen_sec) if batch_gen_sec > 0 else None
            for idx, gen_wavs in zip(batch_indices, gen_wav_list):
                a, b = plans[idx].ranges[rnd]
                state[idx]["prev_wav"] = gen_wavs
                state[idx]["chunk_wavs"].append(gen_wavs)
                chunk_entry = {
                    "round": rnd,
                    "turn_start": a,
                    "turn_end": b,
                    "predicted_sec": round(plans[idx].chunk_secs[rnd], 6),
                    "gen_frames": prepared_inputs[idx]["gen_frames"],
                    "batch_id": batch_id,
                    "batch_size": len(batch_indices),
                    "batch_elapsed_sec": round(float(elapsed), 6),
                    "batch_rtf": batch_rtf,
                }
                if prepared_inputs[idx]["cond_info"]:
                    chunk_entry["conditioning"] = prepared_inputs[idx]["cond_info"]
                state[idx]["chunk_meta"].append(chunk_entry)

    meta_lines: list[str] = []
    wav_lines: list[str] = []
    prompt_lines: list[str] = []
    text_lines: list[str] = []
    mix_lines: list[str] = []
    for idx in my_indices:
        record = records[idx]
        n = record.num_channels
        wid = record.dialogue_id
        st = state[idx]
        gen_full = crossfade_concat(st["chunk_wavs"], round(cross_fade_sec * fs))
        ref_texts = _reference_texts(record.turns, n)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, gen_full[ch], fs)
            write_wav(test_dir / prompt_rel, st["blocks"][ch][ch], fs)
            channels.append(
                {
                    "gen_wav": gen_rel,
                    "prompt_wav": prompt_rel,
                    "ref_text": ref_texts[ch],
                }
            )
            wav_lines.append(f"{wid}_ch{ch} {gen_rel}")
            prompt_lines.append(f"{wid}_ch{ch} {prompt_rel}")
            text_lines.append(f"{wid}_ch{ch} {ref_texts[ch]}")
        mix_rel = f"mix/{wid}.wav"
        write_wav(test_dir / mix_rel, gen_full.sum(dim=0).cpu() / n, fs)
        mix_lines.append(f"{wid} {mix_rel}")

        plan = plans[idx]
        meta = {
            "window_id": wid,
            "session_id": wid,
            "mode": MODE,
            "testset": "covomix2-dialogue-testset",
            "sample_rate": fs,
            "num_channels": n,
            "window_duration_sec": round(gen_full.shape[1] / fs, 6),
            "duration": duration_meta(duration_scale, speed, predicted[idx]),
            "has_reference_audio": False,
            "turn_times": "ordinal",
            # Wall clock is shared across a batch and rounds; per-call batch
            # stats live under chunking.chunks, so no single dialogue-level
            # rtf exists here.
            "rtf": None,
            "compute": {"autocast_dtype": autocast_dtype},
            "chunking": {
                "policy": chunk_cfg,
                "cover_all_speakers": cover_all_speakers,
                "cross_fade_sec": cross_fade_sec,
                "cond_silence_gate": cond.silence_gate,
                "cond_gate_threshold": cond.gate_threshold,
                "cond_gate_fill": cond.gate_fill,
                "cond_loudness_norm": cond.loudness_norm,
                "n_chunks": plan.n_chunks,
                "oversized": list(plan.oversized),
                "chunks": st["chunk_meta"],
            },
            "mix_wav": mix_rel,
            "prompt": {
                "total_sec": round(st["prompt_frames0"] * hop / fs, 6),
                "total_frames": st["prompt_frames0"],
                "fill": prompt_fill,
                "turns": [
                    {
                        "channel": p.channel,
                        "text": p.text,
                        "audio_path": str(p.audio_path),
                        "duration_sec": round(prompt_secs[idx][p.channel], 6),
                    }
                    for p in record.prompts
                ],
            },
            "channels": channels,
            "turns": [
                {
                    "channel": int(t.channel),
                    "text": t.text,
                    "start": t.start,
                    "end": t.end,
                }
                for t in record.turns
            ],
        }
        meta_rel = f"meta/{wid}.json"
        (test_dir / meta_rel).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_lines.append(f"{wid} {meta_rel}")

    suffix = "" if shard_count == 1 else f".{shard_index}of{shard_count}"
    for name, lines in (
        ("meta", meta_lines),
        ("wav", wav_lines),
        ("prompt", prompt_lines),
        ("text", text_lines),
        ("mix", mix_lines),
    ):
        _write_scp(test_dir / f"{name}.scp{suffix}", lines)

    logger.info(
        "chunked external infer done: %d generated in %d batches / %d rounds -> %s%s",
        len(meta_lines),
        n_batches,
        max_rounds,
        test_dir,
        (
            ""
            if shard_count == 1
            else f" (shard {shard_index}/{shard_count}; run "
            f"local/merge_shards.py {test_dir} when all shards are done)"
        ),
    )
    return {
        "n_selected": len(meta_lines),
        "n_skipped": exclusions["n_out_of_band"],
        "n_not_sampled": exclusions["n_not_sampled"],
        "n_other_shards": len(indices) - len(my_indices),
        "n_rounds": max_rounds,
        "n_batches": n_batches,
    }
