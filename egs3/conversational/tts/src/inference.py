"""espnet3 ``infer`` stage for the conversational multi-branch F5 recipe.

Batch-generates multi-channel conversations from manifest windows in three
modes that share ONE output layout so the later ``measure`` stage needs zero
metric-side special-casing:

* ``generate`` - run the assembled model (zero-init gates by default);
* ``gt``       - copy the ground-truth generated-region audio;
* ``resynth``  - round-trip the ground truth through mel + Vocos.

Per window the stage snaps a prompt boundary to an eligible turn boundary,
keeps every channel's audio before it as the acoustic prompt, and infills the
remainder.  Windows with no eligible boundary in the ``[prompt_min,
prompt_max]`` band are skipped and counted; selection is seeded and logged.

Output contract, under ``inference_dir/<test_name>/`` (ALL paths in
``meta.scp`` and in the meta JSONs are relative to THIS directory, so the whole
tree is relocatable):

* ``meta.scp``               - ``<window_id> meta/<window_id>.json``; the
  PRIMARY input every metric iterates.
* ``meta/<window_id>.json``  - prompt boundary (sec + frames), window duration,
  sample rate, per-channel relative paths (generated / prompt / ground-truth
  generated-region wav) and reference text for the generated region, the
  ground-truth turn spans shifted to window time, and RTF (generate mode only).
* convenience SCPs (NOT consumed by metrics): channel-level ``wav.scp`` /
  ``prompt.scp`` / ``text.scp`` (``<window_id>_ch<k>`` rows) and window-level
  ``mix.scp``.

``prompt_boundary_sec`` is the nominal snapped instant; ``prompt_boundary_frames``
is that instant floored to a hop, so the two are quantization-inconsistent by
under one frame.  Metrics that need the exact audio cut (where ``prompt_wav``
ends and ``gt_wav``/``gen_wav`` begin) MUST use ``prompt_boundary_frames * hop``,
not ``prompt_boundary_sec``.

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

from egs3.conversational.tts.dataset.preprocessing.windows import (
    blocked_intervals,
    is_eligible_boundary,
)
from egs3.conversational.tts.src.generation import (
    build_dataset,
    build_preprocessor,
    generate_region,
    load_model,
    load_vocoder,
    pad_branch_text,
    resynth_region,
    write_wav,
)

logger = logging.getLogger(__name__)

_EPS = 1e-6
_MODES = ("generate", "gt", "resynth")


def snap_prompt_boundary(
    turns: Sequence[Any],
    t0: float,
    *,
    target_sec: float,
    prompt_min: float,
    prompt_max: float,
    boundary_guard: float = 0.0,
) -> float | None:
    """Snap the prompt boundary to an eligible turn boundary near ``target_sec``.

    ``turns`` carry absolute-second ``start``/``end`` (session time); ``t0`` is
    the window's absolute start, so a window-relative instant ``rel`` maps to
    absolute ``t0 + rel``.  Candidates are the turns' own endpoints; a candidate
    is eligible iff no turn strictly contains it (the windowing rule, reused via
    ``blocked_intervals`` / ``is_eligible_boundary``).  Among eligible endpoints
    whose window-relative time lies in ``[prompt_min, prompt_max]``, return the
    one closest to ``target_sec`` (ties toward the earlier instant).  Returns
    ``None`` when the band holds no eligible boundary.
    """
    blocked = blocked_intervals(turns, boundary_guard)
    candidates: set[float] = set()
    for turn in turns:
        for edge in (turn.start, turn.end):
            rel = round(edge - t0, 6)
            if prompt_min - _EPS <= rel <= prompt_max + _EPS and is_eligible_boundary(
                blocked, edge
            ):
                candidates.add(rel)
    if not candidates:
        return None
    return min(candidates, key=lambda rel: (abs(rel - target_sec), rel))


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


def _reference_texts(turns, num_channels: int, boundary_rel: float, t0: float):
    """Per-channel generated-region reference text (turns with start >= boundary)."""
    per_channel: list[list[str]] = [[] for _ in range(num_channels)]
    for turn in turns:
        rel_start = round(turn.start - t0, 6)
        if rel_start >= boundary_rel - _EPS:
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


def run_inference(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
) -> dict[str, Any]:
    """Execute the infer stage; return ``{"n_selected", "n_skipped"}``.

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

    if training_config is None:
        train_path = Path(cfg.training_config)
        if not train_path.is_absolute():
            train_path = Path(cfg.get("recipe_dir", ".")) / train_path
        training_config = OmegaConf.load(train_path)

    device = torch.device(cfg.get("device", "cpu"))
    fs = int(training_config.sample_rate)
    hop = int(training_config.hop_length)

    dataset = build_dataset(
        training_config,
        cfg.dataset.split,
        inference=True,
        manifest_path=cfg.dataset.get("manifest_path"),
        dataset_root=cfg.dataset.get("dataset_root"),
    )
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

    prompt_cfg = cfg.prompt
    samp = cfg.sampling
    for idx in indices:
        record = dataset.records[idx]
        boundary_rel = snap_prompt_boundary(
            record.turns,
            record.t0,
            target_sec=float(prompt_cfg.target_sec),
            prompt_min=float(prompt_cfg.min_sec),
            prompt_max=float(prompt_cfg.max_sec),
            boundary_guard=float(prompt_cfg.get("boundary_guard", 0.0)),
        )
        if boundary_rel is None:
            n_skipped += 1
            logger.info("skip %s: no eligible boundary in band", record.window_id)
            continue

        sample = dataset[idx]
        n = sample["num_channels"]
        speech = sample["speech"].to(device)  # (N, T_wav)
        prompt_frames = round(boundary_rel * fs) // hop
        prompt_samples = prompt_frames * hop
        total_frames = speech.shape[1] // hop
        if prompt_frames >= total_frames:
            n_skipped += 1
            logger.info("skip %s: prompt covers the whole window", record.window_id)
            continue

        gt_region = speech[:, prompt_samples:]  # (N, T_gt)
        rtf = None
        if mode == "gt":
            gen_wavs = gt_region.cpu()
        elif mode == "resynth":
            gen_wavs = resynth_region(model, vocoder, gt_region)
        else:  # generate
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

        wid = record.window_id
        ref_texts = _reference_texts(record.turns, n, boundary_rel, record.t0)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            gt_rel = f"gt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, gen_wavs[ch], fs)
            write_wav(test_dir / prompt_rel, speech[ch, :prompt_samples].cpu(), fs)
            write_wav(test_dir / gt_rel, gt_region[ch].cpu(), fs)
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
            "prompt_boundary_sec": boundary_rel,
            "prompt_boundary_frames": prompt_frames,
            "window_duration_sec": round(record.t1 - record.t0, 6),
            "rtf": rtf,
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
        "infer done: %d generated, %d skipped -> %s",
        n_selected,
        n_skipped,
        test_dir,
    )
    return {"n_selected": n_selected, "n_skipped": n_skipped}


def _write_scp(path: Path, lines: Sequence[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
