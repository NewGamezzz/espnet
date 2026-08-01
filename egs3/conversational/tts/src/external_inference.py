"""``infer`` stage variant that generates from an EXTERNAL, audio-free
dialogue test set (currently the public CoVoMix2 set; see
``src/external_testset.py``).

Runs alongside ``src/inference.py`` rather than inside it.  The SSSD path is
not touched, imported into, or parameterized by this module, so the existing
``infer_generate*`` runs stay bit-reproducible; the two share only the
stateless primitives in ``src/generation.py`` and two output-format helpers
imported from ``src/inference.py``.

What differs from the SSSD path
-------------------------------
* **Prompts are external and mono.**  Each speaker's prompt is one
  LibriSpeech utterance, so the prompt block for channel ``k`` carries that
  utterance on row ``k`` and DIGITAL SILENCE on every other row.  Blocks are
  concatenated in channel order exactly as the SSSD path concatenates its
  solo turns, which keeps ``channels[k].prompt_wav`` (row ``k`` of block
  ``k``) the same speaker-similarity reference the metric already expects.
  Silence on the other rows is the one real conditioning difference: the
  SSSD path puts each channel's own (near-silent, real) room audio there.
* **Duration is predicted, not measured** - see
  ``external_testset.estimate_duration_sec``.  The conditioning tensor's
  generated region is therefore ZEROS: there is no ground truth to put
  there, and none is needed (the ODE is conditioned only on the first
  ``prompt_frames``).
* **No ``gt/`` outputs and no ``resynth`` mode.**  This test set ships no
  reference audio, so the ground-truth and vocoder-ceiling anchors are not
  available here; run those on the SSSD split.

Everything else - the ``meta.scp`` contract, the per-channel/mix wav layout,
the reference-text fields - is byte-identical in shape to the SSSD path, so
the entire measure stage runs unchanged.

Reading the results
-------------------
WER, SIM-o and UTMOS are duration-insensitive in the ways that matter and
can be read directly.  The INTERACTION metrics cannot: total duration is the
only timing signal the model receives, so a window predicted too long is
filled with silence, which mechanically moves pause and gap rates.  Treat
interaction numbers from this stage as a function of the duration policy
(recorded in every meta JSON under ``duration``) and measure their
sensitivity with the ``speed`` sweep rather than quoting them as model
properties.  The SSSD split, which has real turn times, remains the place
turn-taking is evaluated.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.src.external_testset import (
    DEFAULT_DURATION_SCALE,
    ExternalRecord,
    duration_meta,
    estimate_duration_sec,
    load_covomix2_testset,
    select_records,
)
from egs3.conversational.tts.src.generation import (
    build_preprocessor,
    generate_region,
    load_model,
    load_vocoder,
    pad_branch_text,
    write_wav,
)

# Output-format helpers, reused verbatim so the two infer paths can never
# drift apart in what they write.  Imported rather than copied; imported
# rather than promoted to public names, so src/inference.py is left untouched.
from egs3.conversational.tts.src.inference import _reference_texts, _write_scp

logger = logging.getLogger(__name__)

MODE = "generate_external"


def _probe_duration_sec(path: Path) -> float:
    """Prompt duration WITHOUT decoding: needed for every dialogue up front
    (duration prediction gates selection), while audio is only decoded for
    the dialogues actually generated."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _load_prompt_wav(path: Path, target_fs: int) -> torch.Tensor:
    """Read a mono prompt utterance and resample to the model rate.

    Returns ``(T,)``.  A multi-channel prompt file is rejected rather than
    silently downmixed: the test set's prompts are single-speaker LibriSpeech
    utterances, and anything else means a mis-pointed root.
    """
    import soundfile as sf

    array, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if array.shape[1] != 1:
        raise ValueError(
            f"{path}: expected a mono prompt, got {array.shape[1]} channels"
        )
    wav = torch.from_numpy(array[:, 0].copy())
    if rate != target_fs:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), orig_freq=rate, new_freq=target_fs
        ).squeeze(0)
    return wav


def _prompt_blocks(
    prompt_wavs: list[torch.Tensor], num_channels: int
) -> list[torch.Tensor]:
    """One full-width ``(N, T_k)`` block per channel: that channel's prompt on
    its own row, silence elsewhere (see the module docstring)."""
    blocks = []
    for ch, wav in enumerate(prompt_wavs):
        block = torch.zeros(num_channels, wav.shape[0], dtype=wav.dtype)
        block[ch] = wav
        blocks.append(block)
    return blocks


def _prompt_turns(record: ExternalRecord) -> list[Turn]:
    """Prompt text as turns, channel-ascending, matching the block order the
    audio was concatenated in."""
    return [
        Turn(
            channel=p.channel,
            speaker=f"prompt_spk{p.channel + 1}",
            text=p.text,
            start=float(i),  # ordinal, not seconds
            end=float(i),
        )
        for i, p in enumerate(record.prompts)
    ]


def run_external_inference(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
) -> dict[str, Any]:
    """Execute the external-test-set infer stage; return counts.

    ``training_config`` / ``model`` / ``vocoder`` are the same injection
    seams ``run_inference`` offers, so tests drive this CPU-only with a tiny
    random-init DiT and a fake vocoder.
    """
    cfg = inference_config
    mode = cfg.get("mode")
    if mode != MODE:
        raise ValueError(f"expected mode {MODE!r}, got {mode!r}")

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
    predicted = [
        estimate_duration_sec(r, secs, duration_scale=duration_scale, speed=speed)
        for r, secs in zip(records, prompt_secs)
    ]
    indices, exclusions = select_records(records, predicted, cfg.selection)
    logger.info(
        "external infer selection: %d/%d dialogues "
        "(%d out of duration band, %d not sampled; scale=%.4f, speed=%.3f)",
        len(indices),
        len(records),
        exclusions["n_out_of_band"],
        exclusions["n_not_sampled"],
        duration_scale,
        speed,
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

    meta_lines: list[str] = []
    wav_lines: list[str] = []
    prompt_lines: list[str] = []
    text_lines: list[str] = []
    mix_lines: list[str] = []
    samp = cfg.sampling

    for idx in tqdm(indices, desc=f"infer[{MODE}]", unit="dialogue"):
        record = records[idx]
        n = record.num_channels

        prompt_wavs = [_load_prompt_wav(p.audio_path, fs) for p in record.prompts]
        blocks = _prompt_blocks(prompt_wavs, n)
        prompt_raw = torch.cat(blocks, dim=1)  # (N, P)
        prompt_frames = prompt_raw.shape[1] // hop
        prompt_samples = prompt_frames * hop
        prompt_trimmed = prompt_raw[:, :prompt_samples]

        gen_frames = max(1, round(predicted[idx] * fs / hop))
        # The generated region is zeros: nothing is known about it, and the
        # ODE conditions only on the first `prompt_frames`.
        speech = torch.cat(
            [prompt_trimmed, torch.zeros(n, gen_frames * hop)], dim=1
        ).to(device)
        total_frames = prompt_frames + gen_frames

        sample = {
            "turns": _prompt_turns(record) + list(record.turns),
            "num_channels": n,
        }
        sample = preprocessor(record.dialogue_id, sample)
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

        wid = record.dialogue_id
        ref_texts = _reference_texts(record.turns, n)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, gen_wavs[ch], fs)
            write_wav(test_dir / prompt_rel, blocks[ch][ch], fs)
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
        write_wav(test_dir / mix_rel, gen_wavs.sum(dim=0).cpu() / n, fs)
        mix_lines.append(f"{wid} {mix_rel}")

        meta = {
            "window_id": wid,
            "session_id": wid,
            "mode": MODE,
            "testset": "covomix2-dialogue-testset",
            "sample_rate": fs,
            "num_channels": n,
            "window_duration_sec": round(gen_frames * hop / fs, 6),
            "duration": duration_meta(duration_scale, speed, predicted[idx]),
            "has_reference_audio": False,
            "turn_times": "ordinal",
            "rtf": rtf,
            "mix_wav": mix_rel,
            "prompt": {
                "total_sec": round(prompt_samples / fs, 6),
                "total_frames": prompt_frames,
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

    _write_scp(test_dir / "meta.scp", meta_lines)
    _write_scp(test_dir / "wav.scp", wav_lines)
    _write_scp(test_dir / "prompt.scp", prompt_lines)
    _write_scp(test_dir / "text.scp", text_lines)
    _write_scp(test_dir / "mix.scp", mix_lines)

    logger.info("external infer done: %d generated -> %s", len(meta_lines), test_dir)
    # n_skipped keeps the SSSD path's meaning - "could not be generated" -
    # so it counts ONLY out-of-band dialogues; deliberately not-sampled ones
    # are reported separately and never inflate a failure-shaped number.
    return {
        "n_selected": len(meta_lines),
        "n_skipped": exclusions["n_out_of_band"],
        "n_not_sampled": exclusions["n_not_sampled"],
    }
