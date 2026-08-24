"""``ingest_external_system``: score ANOTHER system's audio with our measure
stage.

The cross-model comparison (see the two-tier evaluation framework) re-runs
every baseline ourselves rather than quoting published numbers.  Those
systems have their own repos, environments and checkpoints, so their
inference happens entirely OUTSIDE this recipe; what comes back is a
directory of wavs named by dialogue id.  This mode is the adapter: it reads
that directory against the same external manifest our own runs read, and
writes the SAME output contract ``src/external_anchor.py`` writes (meta /
wav / prompt / mix / gt SCPs, the same meta keys), so the ordinary measure
stage scores baseline and system with one identical pipeline.

Nothing here runs a model.  No GPU, no vocoder: read, split, resample, write.

What it assumes about the baseline's output
-------------------------------------------
* One file per dialogue, ``<wav_dir>/<dialogue_id><suffix>`` (default
  ``.wav``), with ONE TRACK PER SPEAKER in channel order - the shape a
  multi-channel system emits (e.g. ZipVoice-Dialog-Stereo).  A file with
  more tracks than the record has channels is an error, not a downmix: a
  silent mis-mapping would corrupt every per-channel number.
* A record with ``num_channels == 1`` takes track 0 and IGNORES the rest;
  ``ingest.mono_extra_track`` decides whether a monologue row's second
  track must be QUIET (``require_silent``), may carry anything
  (``ignore``), or is an error (``forbid``).  Quiet is relative, not
  absolute: a generative model never writes digital silence, so the test
  is that the unused track sits at least ``ingest.mono_extra_track_db``
  below the record's own speech (measured on ZipVoice-Dialog-Stereo:
  35-63 dB below, i.e. inaudible, but nowhere near zero).
* Any sample rate: it is resampled to the training rate, like every other
  path here.

Mono (single-track) baselines: ``ingest.output: mixture``
---------------------------------------------------------
A mixture-only system (e.g. the mono ZipVoice-Dialog) has no per-channel
truth, so most of the battery does not apply to it.  Rather than inventing
channels with a diarizer - which would make its rows incomparable to the
channel systems' - this mode writes the mixture as a ONE-entry record whose
reference is the whole conversation in turn order, and DELIBERATELY OMITS
``prompt_wav`` and ``gt_wav`` from that entry.  The omission is the safety
rail: SpeakerSimilarityMetric and InteractionMetric raise on the missing
keys instead of quietly returning a number that compares a two-speaker
mixture against one speaker's prompt.

What such an arm can be quoted on: ``wer_mix`` and ``utmos_mix``.
``wer_channel`` is computed but is a DUPLICATE of ``wer_mix`` there (same
audio, same reference), and ``utmos_ipu`` is UTMOS over mixture IPUs, which
contain overlapping speech - read neither as a per-channel number.  Run it
with a reduced metrics config (ASR + quality only).

Channel mapping is VERIFIED, not assumed
----------------------------------------
Which track carries which speaker is a property of the baseline's own
inference, and getting it backwards silently swaps every ``wer_channel``,
``sim_o`` and ``*_dur_w1`` row.  ``ingest.verify_channel_map`` therefore
re-derives the assignment per dialogue from first-onset order and records
the outcome in the meta (``channel_map``: ``as_is`` / ``swapped`` /
``unverifiable``) and in the stage's return counts.  It never re-orders
audio on its own: a disagreement is reported so the run can be re-read,
because the honest fix for a systematically swapped baseline is a
``channel_order`` config, not a per-row guess.

Reading the results
-------------------
A baseline that predicts its own durations has no duration oracle, so its
meta carries ``duration.source == "system"`` and its rows compare ONLY to
our predicted-duration arms - never to a ``ground_truth`` (gtdur) arm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from egs3.conversational.tts.src.external_inference import (
    _load_prompt_wav,
    _probe_duration_sec,
)
from egs3.conversational.tts.src.external_testset import (
    duration_meta,
    load_records,
    select_records,
)
from egs3.conversational.tts.src.generation import write_wav
from egs3.conversational.tts.src.inference import _reference_texts, _write_scp

logger = logging.getLogger(__name__)

MODE = "ingest_external_system"

OUTPUT_KINDS = ("channels", "mixture")
MONO_EXTRA_TRACK_POLICIES = ("require_silent", "ignore", "forbid")
#: How far below the record's own speech an unused track has to sit before
#: it counts as silence, in dB.  An absolute floor would be the wrong test:
#: what matters is that nothing audible was put where the transcript has no
#: speaker, and 30 dB down is already inaudible against the other channel.
DEFAULT_MONO_EXTRA_TRACK_DB = 30.0
#: Onset detection for the channel-map check: the first frame whose RMS
#: crosses this, over 100 ms frames - the same shape of test the test-set
#: builder used to map the reference tracks.
ONSET_FRAME_SEC = 0.1
ONSET_RMS_THRESHOLD = 0.01


def _load_multichannel(path: Path, target_fs: int) -> torch.Tensor:
    """Read a wav as ``(C, T)`` at ``target_fs``."""
    import soundfile as sf
    import torchaudio

    array, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(array.copy()).transpose(0, 1).contiguous()
    if rate != target_fs:
        wav = torchaudio.functional.resample(wav, orig_freq=rate, new_freq=target_fs)
    return wav


def _rms_db(wav: torch.Tensor) -> float:
    """Full-signal RMS in dBFS; ``-inf``-safe."""
    return float(20.0 * torch.log10(wav.pow(2).mean().sqrt() + 1e-12))


def _quiet_margin_db(used: list[torch.Tensor], extra: list[torch.Tensor]) -> float:
    """How far the loudest unused track sits below the loudest used one."""
    return max(_rms_db(track) for track in used) - max(
        _rms_db(track) for track in extra
    )


def _onset_index(wav: torch.Tensor, fs: int) -> int | None:
    """Index of the first loud 100 ms frame, or ``None`` if never loud."""
    frame = max(1, int(round(ONSET_FRAME_SEC * fs)))
    n_frames = wav.shape[0] // frame
    if n_frames == 0:
        return None
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    rms = frames.pow(2).mean(dim=1).sqrt()
    loud = (rms > ONSET_RMS_THRESHOLD).nonzero()
    return int(loud[0].item()) if loud.numel() else None


def _first_speaking_channel(turns) -> int | None:
    """Channel of the first turn, i.e. who the transcript says speaks first."""
    for turn in sorted(turns, key=lambda t: t.start):
        return int(turn.channel)
    return None


def _channel_map_verdict(tracks: list[torch.Tensor], turns, fs: int) -> str:
    """``as_is`` / ``swapped`` / ``unverifiable`` for a 2-channel dialogue.

    Compares the transcript's first speaker against the track that starts
    first.  Unverifiable when a track never crosses the onset threshold, or
    when both start in the same frame - in which case the ordering carries
    no information and claiming either answer would be an invention.
    """
    expected = _first_speaking_channel(turns)
    if expected is None:
        return "unverifiable"
    onsets = [_onset_index(track, fs) for track in tracks]
    if any(onset is None for onset in onsets):
        return "unverifiable"
    if onsets[0] == onsets[1]:
        return "unverifiable"
    observed = int(min(range(len(onsets)), key=lambda i: onsets[i]))
    return "as_is" if observed == expected else "swapped"


def _write_mixture_record(
    test_dir: Path,
    lines: dict[str, list[str]],
    record,
    mixture: torch.Tensor,
    fs: int,
    *,
    system: dict[str, Any],
    testset_name: str,
) -> None:
    """Write a mono system's dialogue as a one-entry, mixture-level record.

    The single entry carries ``gen_wav`` and ``ref_text`` and NOTHING else:
    a mixture has no prompt to be similar to and no per-channel reference to
    be timed against, so the per-channel metrics must fail loudly rather
    than return a number nobody can interpret.
    """
    wid = record.dialogue_id
    ref_text = " ".join(
        turn.text for turn in sorted(record.turns, key=lambda t: t.start)
    )
    gen_rel = f"wav/{wid}_ch0.wav"
    mix_rel = f"mix/{wid}.wav"
    write_wav(test_dir / gen_rel, mixture, fs)
    write_wav(test_dir / mix_rel, mixture, fs)
    lines["wav"].append(f"{wid}_ch0 {gen_rel}")
    lines["text"].append(f"{wid}_ch0 {ref_text}")
    lines["mix"].append(f"{wid} {mix_rel}")

    generated_sec = mixture.shape[0] / fs
    prompt_secs = [_probe_duration_sec(p.audio_path) for p in record.prompts]
    meta = {
        "window_id": wid,
        "session_id": wid,
        "mode": MODE,
        "testset": testset_name,
        "system": system,
        "output": "mixture",
        "sample_rate": fs,
        # The OUTPUT has one channel; the record's speaker count is kept
        # separately so no reader mistakes this for a monologue.
        "num_channels": 1,
        "record_num_channels": record.num_channels,
        "window_duration_sec": round(generated_sec, 6),
        "duration": duration_meta(
            1.0, 1.0, generated_sec, source="system", gt_sec=record.gt_duration_sec
        ),
        "has_reference_audio": False,
        "gt_duration_sec": record.gt_duration_sec,
        "turn_times": "ordinal",
        "rtf": None,
        "mix_wav": mix_rel,
        "channel_map": "mixture",
        "prompt": {
            "total_sec": round(sum(prompt_secs), 6),
            "turns": [
                {
                    "channel": p.channel,
                    "text": p.text,
                    "audio_path": str(p.audio_path),
                    "duration_sec": round(prompt_secs[p.channel], 6),
                }
                for p in record.prompts
            ],
        },
        "channels": [{"gen_wav": gen_rel, "ref_text": ref_text}],
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
    lines["meta"].append(f"{wid} {meta_rel}")


def run_external_system_ingest(
    inference_config, *, training_config=None
) -> dict[str, Any]:
    """Ingest a baseline's per-dialogue wavs; return counts."""
    cfg = inference_config
    mode = cfg.get("mode")
    if mode != MODE:
        raise ValueError(f"expected mode {MODE!r}, got {mode!r}")

    if training_config is None:
        train_path = Path(cfg.training_config)
        if not train_path.is_absolute():
            train_path = Path(cfg.get("recipe_dir", ".")) / train_path
        training_config = OmegaConf.load(train_path)
    fs = int(training_config.sample_rate)

    ingest = cfg.get("ingest", {}) or {}
    system_cfg = cfg.get("system", None)
    system = (
        OmegaConf.to_container(system_cfg, resolve=True)
        if OmegaConf.is_config(system_cfg)
        else dict(system_cfg or {})
    )
    if not system.get("name"):
        raise ValueError("ingest_external_system needs system.name for provenance")
    wav_dir = Path(ingest.get("wav_dir", ""))
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"ingest.wav_dir does not exist: {wav_dir}")
    suffix = str(ingest.get("suffix", ".wav"))
    output_kind = str(ingest.get("output", "channels"))
    if output_kind not in OUTPUT_KINDS:
        raise ValueError(
            f"ingest.output must be one of {OUTPUT_KINDS}, got {output_kind!r}"
        )
    mono_policy = str(ingest.get("mono_extra_track", "require_silent"))
    if mono_policy not in MONO_EXTRA_TRACK_POLICIES:
        raise ValueError(
            f"ingest.mono_extra_track must be one of {MONO_EXTRA_TRACK_POLICIES}, "
            f"got {mono_policy!r}"
        )
    extra_track_db = float(
        ingest.get("mono_extra_track_db", DEFAULT_MONO_EXTRA_TRACK_DB)
    )
    verify_map = bool(ingest.get("verify_channel_map", True))
    channel_order = ingest.get("channel_order", None)

    token_list = OmegaConf.to_container(training_config, resolve=True)["dataset"][
        "preprocessor"
    ]["token_list"]
    records, testset_name = load_records(cfg.testset, token_list)
    gt_secs = [float(r.gt_duration_sec) for r in records]
    selection = cfg.get("selection", {}) or {}
    indices, exclusions = select_records(records, gt_secs, selection)

    missing = [
        records[i].dialogue_id
        for i in indices
        if not (wav_dir / f"{records[i].dialogue_id}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(indices)} selected dialogue(s) have no "
            f"{system['name']} output in {wav_dir}, e.g. {missing[:5]}"
        )
    logger.info(
        "ingest %s: %d/%d dialogues of %s from %s",
        system["name"],
        len(indices),
        len(records),
        testset_name,
        wav_dir,
    )

    test_dir = Path(cfg.inference_dir) / cfg.test_name
    for sub in ("meta", "wav", "prompt", "mix", "gt"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    lines: dict[str, list[str]] = {
        k: [] for k in ("meta", "wav", "prompt", "text", "mix", "gt")
    }
    verdicts: dict[str, int] = {"as_is": 0, "swapped": 0, "unverifiable": 0}
    for idx in indices:
        record = records[idx]
        n = record.num_channels
        wid = record.dialogue_id
        ref_texts = _reference_texts(record.turns, n)

        gen = _load_multichannel(wav_dir / f"{wid}{suffix}", fs)

        if output_kind == "mixture":
            if gen.shape[0] != 1:
                raise ValueError(
                    f"{wid}: ingest.output='mixture' expects a single-track file, "
                    f"got {gen.shape[0]} tracks"
                )
            _write_mixture_record(
                test_dir,
                lines,
                record,
                gen[0],
                fs,
                system=system,
                testset_name=testset_name,
            )
            verdicts["unverifiable"] += 1
            continue

        if gen.shape[0] < n:
            raise ValueError(
                f"{wid}: {system['name']} wrote {gen.shape[0]} track(s) but the "
                f"record has {n} channel(s)"
            )
        tracks = [gen[c] for c in range(gen.shape[0])]
        if channel_order is not None:
            order = [int(c) for c in channel_order]
            if sorted(order) != list(range(len(tracks))):
                raise ValueError(
                    f"ingest.channel_order must be a permutation of "
                    f"0..{len(tracks) - 1}, got {list(channel_order)}"
                )
            tracks = [tracks[c] for c in order]

        extra = tracks[n:]
        if extra and mono_policy == "forbid":
            raise ValueError(
                f"{wid}: {gen.shape[0]} tracks for a {n}-channel record and "
                f"ingest.mono_extra_track='forbid'"
            )
        if extra and mono_policy == "require_silent":
            margin = _quiet_margin_db(tracks[:n], extra)
            if margin < extra_track_db:
                raise ValueError(
                    f"{wid}: the unused track(s) of a {n}-channel record sit only "
                    f"{margin:.1f} dB below its speech (< {extra_track_db:.1f} dB); "
                    f"the baseline put something audible where this record has no "
                    f"speaker"
                )
        tracks = tracks[:n]

        verdict = "unverifiable"
        if verify_map and n == 2:
            verdict = _channel_map_verdict(tracks, record.turns, fs)
        verdicts[verdict] += 1

        length = min(track.shape[0] for track in tracks)
        gt_wavs = (
            [_load_prompt_wav(p, fs) for p in record.gt_paths]
            if record.gt_paths is not None
            else None
        )
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, tracks[ch][:length], fs)
            write_wav(
                test_dir / prompt_rel,
                _load_prompt_wav(record.prompts[ch].audio_path, fs),
                fs,
            )
            entry = {
                "gen_wav": gen_rel,
                "prompt_wav": prompt_rel,
                "ref_text": ref_texts[ch],
            }
            if gt_wavs is not None:
                gt_rel = f"gt/{wid}_ch{ch}.wav"
                write_wav(test_dir / gt_rel, gt_wavs[ch], fs)
                entry["gt_wav"] = gt_rel
                lines["gt"].append(f"{wid}_ch{ch} {gt_rel}")
            channels.append(entry)
            lines["wav"].append(f"{wid}_ch{ch} {gen_rel}")
            lines["prompt"].append(f"{wid}_ch{ch} {prompt_rel}")
            lines["text"].append(f"{wid}_ch{ch} {ref_texts[ch]}")
        mix_rel = f"mix/{wid}.wav"
        mix = sum(track[:length] for track in tracks) / n
        write_wav(test_dir / mix_rel, mix, fs)
        lines["mix"].append(f"{wid} {mix_rel}")

        prompt_secs = [_probe_duration_sec(p.audio_path) for p in record.prompts]
        generated_sec = length / fs
        meta = {
            "window_id": wid,
            "session_id": wid,
            "mode": MODE,
            "testset": testset_name,
            "system": system,
            "sample_rate": fs,
            "num_channels": n,
            "window_duration_sec": round(generated_sec, 6),
            # The baseline chose this length itself; there is no rule and no
            # oracle, so predicted_sec IS what it produced.  Recorded as
            # source "system" so no reader mistakes these rows for a
            # duration-oracle arm.
            "duration": duration_meta(
                1.0,
                1.0,
                generated_sec,
                source="system",
                gt_sec=record.gt_duration_sec,
            ),
            "has_reference_audio": gt_wavs is not None,
            "gt_duration_sec": record.gt_duration_sec,
            "turn_times": "ordinal",
            "rtf": None,
            "mix_wav": mix_rel,
            "channel_map": verdict,
            "prompt": {
                "total_sec": round(sum(prompt_secs), 6),
                "turns": [
                    {
                        "channel": p.channel,
                        "text": p.text,
                        "audio_path": str(p.audio_path),
                        "duration_sec": round(prompt_secs[p.channel], 6),
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
        lines["meta"].append(f"{wid} {meta_rel}")

    for name, scp_lines in lines.items():
        _write_scp(test_dir / f"{name}.scp", scp_lines)
    if verdicts["swapped"]:
        logger.warning(
            "channel map: %d dialogue(s) start on the track the transcript does "
            "NOT expect; check ingest.channel_order before quoting per-channel "
            "numbers",
            verdicts["swapped"],
        )
    logger.info(
        "ingest %s done: %d dialogues -> %s (channel map %s)",
        system["name"],
        len(lines["meta"]),
        test_dir,
        verdicts,
    )
    return {
        "n_selected": len(lines["meta"]),
        "n_skipped": exclusions["n_out_of_band"] + exclusions["n_not_sampled"],
        "channel_map": verdicts,
    }
