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
* **Dialogues are BATCHED into one ODE call** (``cfg.batching``): the packed
  ``counts`` layout the model trains with carries several dialogues at once,
  which is where the runtime went - sequentially, every 32-step CFG ODE ran
  the DiT with only one dialogue's channels as rows.  Batches are planned
  over the full selection and shards take whole batches, so outputs are
  invariant to ``shard_count``; noise inside a multi-dialogue batch depends
  on batch composition (the per-batch reseed generalizes the sequential
  path's per-dialogue reseed), so changing the batching knobs redraws
  equally-valid samples.  ``batching: {}`` (or null budgets) reproduces the
  sequential run bit-for-bit.

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
    assign_shard,
    duration_meta,
    estimate_duration_sec,
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
    shard_count = int(cfg.selection.get("shard_count", 1) or 1)
    shard_index = int(cfg.selection.get("shard_index", 0) or 0)
    indices, exclusions = select_records(records, predicted, cfg.selection)

    # Batches are planned over the FULL selection, then shards take whole
    # batches: batch composition (and with it every dialogue's noise draw)
    # is a pure function of the config, never of shard membership, so the
    # union of all shards is bit-identical to an unsharded run.
    batching = cfg.get("batching", {}) or {}
    max_batch_audio_sec = batching.get("max_batch_audio_sec")
    max_batch_dialogues = batching.get("max_batch_dialogues")
    # The ODE integrates prompt + generated region, padded to the batch's
    # longest dialogue - that total, not the generated region alone, is what
    # the batch budget must price.
    total_secs = [sum(secs) + pred for secs, pred in zip(prompt_secs, predicted)]
    batches = plan_batches(
        indices,
        total_secs,
        max_batch_audio_sec=(
            float(max_batch_audio_sec) if max_batch_audio_sec is not None else None
        ),
        max_batch_dialogues=(
            int(max_batch_dialogues) if max_batch_dialogues is not None else None
        ),
    )
    batch_costs = [len(b) * max(total_secs[i] for i in b) for b in batches]
    shard_batch_ids = assign_shard(
        list(range(len(batches))), batch_costs, shard_index, shard_count
    )
    n_mine = sum(len(batches[b]) for b in shard_batch_ids)
    logger.info(
        "external infer selection: %d/%d dialogues (%d out of duration band, "
        "%d not sampled, %d other shards; scale=%.4f, speed=%.3f) in %d/%d "
        "batches (budget %s s padded audio, cap %s dialogues)",
        n_mine,
        len(records),
        exclusions["n_out_of_band"],
        exclusions["n_not_sampled"],
        len(indices) - n_mine,
        duration_scale,
        speed,
        len(shard_batch_ids),
        len(batches),
        max_batch_audio_sec,
        max_batch_dialogues,
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
    autocast_dtype = samp.get("autocast_dtype")

    progress = tqdm(
        shard_batch_ids, desc=f"infer[{MODE}]", unit="batch", total=len(shard_batch_ids)
    )
    for batch_id in progress:
        batch_indices = batches[batch_id]
        prepared: list[dict[str, Any]] = []
        for idx in batch_indices:
            record = records[idx]
            n = record.num_channels

            prompt_wavs = [_load_prompt_wav(p.audio_path, fs) for p in record.prompts]
            blocks = _prompt_blocks(prompt_wavs, n)
            prompt_raw = torch.cat(blocks, dim=1)  # (N, P)
            prompt_frames = prompt_raw.shape[1] // hop
            prompt_samples = prompt_frames * hop
            prompt_trimmed = prompt_raw[:, :prompt_samples]

            gen_frames = max(1, round(predicted[idx] * fs / hop))
            # The generated region is zeros: nothing is known about it, and
            # the ODE conditions only on the first `prompt_frames`.
            speech = torch.cat(
                [prompt_trimmed, torch.zeros(n, gen_frames * hop)], dim=1
            ).to(device)

            sample = {
                "turns": _prompt_turns(record) + list(record.turns),
                "num_channels": n,
            }
            sample = preprocessor(record.dialogue_id, sample)
            text = pad_branch_text(sample, device)

            prepared.append(
                {
                    "idx": idx,
                    "record": record,
                    "blocks": blocks,
                    "prompt_frames": prompt_frames,
                    "prompt_samples": prompt_samples,
                    "gen_frames": gen_frames,
                    "item": GenerationItem(
                        speech=speech,
                        text=text,
                        prompt_frames=prompt_frames,
                        total_frames=prompt_frames + gen_frames,
                    ),
                }
            )

        gen_wav_list, elapsed = generate_batch(
            model,
            vocoder,
            [p["item"] for p in prepared],
            steps=int(samp.steps),
            cfg_strength=float(samp.cfg_strength),
            sway_sampling_coef=float(samp.sway_sampling_coef),
            seed=samp.get("seed"),
            autocast_dtype=autocast_dtype,
        )
        batch_gen_sec = sum(w.shape[1] for w in gen_wav_list) / fs
        # Wall clock is spent on the batch as a whole, so RTF is a batch
        # quantity; every member records the same value (plus the batch
        # context under "compute") rather than a fabricated per-dialogue
        # split of the elapsed time.
        rtf = float(elapsed / batch_gen_sec) if batch_gen_sec > 0 else None

        for prep, gen_wavs in zip(prepared, gen_wav_list):
            idx = prep["idx"]
            record = prep["record"]
            blocks = prep["blocks"]
            n = record.num_channels

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
                "window_duration_sec": round(prep["gen_frames"] * hop / fs, 6),
                "duration": duration_meta(duration_scale, speed, predicted[idx]),
                "has_reference_audio": False,
                "turn_times": "ordinal",
                "rtf": rtf,
                "compute": {
                    # The plan-level batch id, shard-invariant by design.
                    "batch_id": batch_id,
                    "batch_size": len(prepared),
                    "batch_elapsed_sec": round(float(elapsed), 6),
                    "autocast_dtype": autocast_dtype,
                },
                "mix_wav": mix_rel,
                "prompt": {
                    "total_sec": round(prep["prompt_samples"] / fs, 6),
                    "total_frames": prep["prompt_frames"],
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

    # Shards share the wav/prompt/mix/meta subdirectories safely - every
    # filename is keyed by the unique dialogue id - but each writes its OWN
    # SCPs, because an SCP is written wholesale and siblings would clobber
    # each other. `local/merge_shards.py` concatenates them once every shard
    # has finished; an unsharded run writes the plain names directly and
    # needs no merge.
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
        "external infer done: %d generated -> %s%s",
        len(meta_lines),
        test_dir,
        (
            ""
            if shard_count == 1
            else f" (shard {shard_index}/{shard_count}; run "
            f"local/merge_shards.py {test_dir} when all shards are done)"
        ),
    )
    # n_skipped keeps the SSSD path's meaning - "could not be generated" - so
    # it counts ONLY out-of-band dialogues; not-sampled and other-shard
    # dialogues are reported separately and never inflate a failure-shaped
    # number.
    return {
        "n_selected": len(meta_lines),
        "n_skipped": exclusions["n_out_of_band"],
        "n_not_sampled": exclusions["n_not_sampled"],
        "n_other_shards": len(indices) - n_mine,
        "n_batches": len(shard_batch_ids),
    }
