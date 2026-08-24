"""``generate_external_gt``: the ground-truth anchor for external dialogue
test sets that ship reference audio (training-style manifests, e.g.
ZipVoice-Dialog test-en).

The SSSD path has ``gt`` / ``resynth`` anchor modes; the CoVoMix2 path could
not, having no reference audio.  This mode writes each dialogue's
ground-truth channels AS the generation - same output contract as
``generate_external_chunked`` (meta / wav / prompt / mix / gt SCPs, the same
meta keys), so the ordinary measure stage scores it unchanged:

* ``wer_channel`` / ``wer_mix`` on the anchor = transcript-vs-ASR
  disagreement, i.e. the reference transcripts' own noise floor - the number
  every system's WER on this set has to be read against.
* ``utmos_*`` / ``sim_o`` on the anchor = the quality/similarity ceiling
  real speech reaches under these metrics.
* every ``*_dur_w1`` collapses to ~0 (gen == gt), which doubles as the
  acceptance test of the ``gt_wav`` plumbing the chunked path writes.

No model, no vocoder, no GPU: a copy with resampling to the training rate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

MODE = "generate_external_gt"


def run_external_gt(inference_config, *, training_config=None) -> dict[str, Any]:
    """Write the ground truth as the generation; return counts."""
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

    token_list = OmegaConf.to_container(training_config, resolve=True)["dataset"][
        "preprocessor"
    ]["token_list"]
    records, testset_name = load_records(cfg.testset, token_list)
    missing = [r.dialogue_id for r in records if r.gt_paths is None]
    if missing:
        raise ValueError(
            f"{len(missing)} dialogue(s) have no ground-truth audio, e.g. "
            f"{missing[:5]}; the gt anchor needs a reference on every dialogue"
        )
    gt_secs = [float(r.gt_duration_sec) for r in records]
    selection = cfg.get("selection", {}) or {}
    indices, exclusions = select_records(records, gt_secs, selection)
    logger.info(
        "gt anchor: %d/%d dialogues of %s", len(indices), len(records), testset_name
    )

    test_dir = Path(cfg.inference_dir) / cfg.test_name
    for sub in ("meta", "wav", "prompt", "mix", "gt"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    lines: dict[str, list[str]] = {
        k: [] for k in ("meta", "wav", "prompt", "text", "mix", "gt")
    }
    for idx in indices:
        record = records[idx]
        n = record.num_channels
        wid = record.dialogue_id
        ref_texts = _reference_texts(record.turns, n)
        gt_wavs = [_load_prompt_wav(p, fs) for p in record.gt_paths]
        length = min(w.shape[0] for w in gt_wavs)
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            gt_rel = f"gt/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            wav = gt_wavs[ch][:length]
            write_wav(test_dir / gen_rel, wav, fs)
            write_wav(test_dir / gt_rel, wav, fs)
            write_wav(
                test_dir / prompt_rel,
                _load_prompt_wav(record.prompts[ch].audio_path, fs),
                fs,
            )
            channels.append(
                {
                    "gen_wav": gen_rel,
                    "prompt_wav": prompt_rel,
                    "gt_wav": gt_rel,
                    "ref_text": ref_texts[ch],
                }
            )
            lines["wav"].append(f"{wid}_ch{ch} {gen_rel}")
            lines["gt"].append(f"{wid}_ch{ch} {gt_rel}")
            lines["prompt"].append(f"{wid}_ch{ch} {prompt_rel}")
            lines["text"].append(f"{wid}_ch{ch} {ref_texts[ch]}")
        mix_rel = f"mix/{wid}.wav"
        mix = sum(w[:length] for w in gt_wavs) / n
        write_wav(test_dir / mix_rel, mix, fs)
        lines["mix"].append(f"{wid} {mix_rel}")

        prompt_secs = [_probe_duration_sec(p.audio_path) for p in record.prompts]
        meta = {
            "window_id": wid,
            "session_id": wid,
            "mode": MODE,
            "testset": testset_name,
            "sample_rate": fs,
            "num_channels": n,
            "window_duration_sec": round(length / fs, 6),
            # The anchor generates nothing; the reference length IS the
            # duration.  ``predicted_sec`` is left equal to it so the
            # ratio reads 1.0 rather than inventing a rule estimate here.
            "duration": duration_meta(
                1.0,
                1.0,
                record.gt_duration_sec,
                source="ground_truth",
                gt_sec=record.gt_duration_sec,
            ),
            "has_reference_audio": True,
            "gt_duration_sec": record.gt_duration_sec,
            "turn_times": "ordinal",
            "rtf": None,
            "mix_wav": mix_rel,
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
    logger.info("gt anchor done: %d dialogues -> %s", len(lines["meta"]), test_dir)
    return {
        "n_selected": len(lines["meta"]),
        "n_skipped": exclusions["n_out_of_band"] + exclusions["n_not_sampled"],
    }
