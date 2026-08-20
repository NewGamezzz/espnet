"""Concatenated single-speaker baseline: stock F5-TTS, one turn at a time.

The comparison our system needs is against a system with NO dialogue
modelling at all: generate every turn independently with ordinary
single-speaker TTS, then concatenate the turns back-to-back.  Whatever our
model buys - overlap, timing, cross-speaker conditioning - has to show up as
a difference against this.

Why the engine is the recipe's own assembled model rather than an external
F5 install
-----------------------------------------------------------------------
``build_multibranch_f5`` with ``ckpt: null`` is the pretrained F5TTS_Base
checkpoint with zero-init exchange gates, and ``tests/test_pretrained_real
.py::test_single_channel_sample_parity_vs_baseline_cfm`` pins the fact that
``counts=[1]`` plus zero gates IS the baseline espnet2 CFM - stock F5, to
within bit-stable math.  Every call here is single-channel, so every call is
stock F5.

That choice is deliberate: mel front-end, vocoder, sampler, vocab and
duration rule are then IDENTICAL to the system under test, so the only
difference left is the dialogue modelling.  Wiring in a separate F5
installation would have re-introduced all of those as confounds and made any
gap unattributable.

What differs from the conversational path
-----------------------------------------
* **Text is plain.**  No ``<turn>``, no ``<OTHER>`` - those tokens are the
  conversational machinery, and a baseline that saw them would not be a
  baseline.  Each call's text is ``ref_text + " " + turn_text`` in plain
  characters, which is exactly what F5's own ``infer_process`` conditions on.
  The vocab is still the extended one; its two extra rows are simply never
  emitted, so the tokenizer stays identical.
* **Conditioning is stateless.**  Every turn of speaker k uses that
  speaker's ONE reference prompt.  No generated audio is ever fed back, so
  the baseline has no error accumulation to accumulate - which is the point.
* **The timeline is a concatenation.**  Turn t starts exactly where turn
  t-1 ended.  Channel k carries its own turns and DIGITAL SILENCE elsewhere;
  the mixdown is the sum.
  Consequence to state whenever these runs are reported: overlap and gap are
  exactly 0 per minute BY CONSTRUCTION.  Those are not scores this baseline
  did badly on, they are structural facts about what a concatenative system
  can express.

Duration
--------
Each source mirrors the duration rule the system under test uses on the same
data, so pacing is never the confound:

* CoVoMix2 has no reference audio, so turn lengths are PREDICTED from the
  speaker's own prompt char-rate (``estimate_turn_secs``) exactly as the
  chunked path predicts them.
* SSSD has reference audio, and the conversational path masks a region equal
  to the full ground-truth window, so per-turn lengths here are the REAL
  turn durations.

Output contract is byte-for-byte the layout of the other infer modes
(``meta.scp`` + ``meta/``, ``wav/``, ``prompt/``, ``mix/``, plus the
convenience SCPs), so ``measure`` runs unchanged and the numbers land on the
same scales as every other row.

Sharding is by dialogue via ``selection.shard_count`` / ``shard_index``.
Turns are generated one ODE call at a time; batching them is a speed
optimization this module deliberately does not do yet, because a wrong batch
grouping would change the noise draw per turn and quietly break comparability
with an unsharded run.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from egs3.conversational.tts.dataset.preprocessing.text import (
    encode_tokens,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab
from egs3.conversational.tts.src.external_inference import _load_prompt_wav
from egs3.conversational.tts.src.external_testset import (
    assign_shard,
    estimate_duration_sec,
    load_covomix2_testset,
    select_records,
)
from egs3.conversational.tts.src.generation import (
    build_dataset,
    generate_region,
    load_model,
    load_vocoder,
    read_audio_span,
    write_wav,
)
from egs3.conversational.tts.src.inference import (
    _build_turn_pools,
    _reference_texts,
    _resolve_pinned_turns,
    _select_indices,
    _write_scp,
)

logger = logging.getLogger(__name__)

MODE = "generate_concat_baseline"

DEFAULT_DURATION_SCALE = 1.117


@dataclass
class BaselineItem:
    """One dialogue, flattened into what a single-speaker TTS system needs.

    ``turn_secs`` is the intended duration of each turn; see the module
    docstring for why its provenance differs by source.
    """

    dialogue_id: str
    num_channels: int
    turn_channels: list[int]
    turn_texts: list[str]
    turn_secs: list[float]
    prompt_wavs: list[torch.Tensor]  # per channel, mono (T,)
    prompt_texts: list[str]
    prompt_secs: list[float]
    extra_meta: dict[str, Any] = field(default_factory=dict)


def plain_text_ids(
    ref_text: str, gen_text: str, token2id: dict[str, int]
) -> torch.Tensor:
    """Token ids for one stock-F5 call, shaped ``(1, L)``.

    F5 conditions on the reference transcript followed by the text to
    generate, as one character sequence - no turn markers, no per-branch
    filler.  Joined with a single space, matching ``infer_process``.
    """
    joined = f"{ref_text} {gen_text}" if ref_text else gen_text
    ids = encode_tokens(list(joined), token2id)
    return torch.tensor([ids], dtype=torch.long)


def concat_timeline(
    turn_wavs: Sequence[torch.Tensor],
    turn_channels: Sequence[int],
    num_channels: int,
) -> torch.Tensor:
    """Lay per-turn mono waves back-to-back on ``num_channels`` rows.

    Turn t occupies ``[offset, offset + len(t))`` on its own channel and every
    other channel is digitally silent there, so the result has zero overlap
    and zero gap by construction.  Returns ``(num_channels, T_total)``.
    """
    if len(turn_wavs) != len(turn_channels):
        raise ValueError(
            f"{len(turn_wavs)} waves for {len(turn_channels)} turn channels"
        )
    if not turn_wavs:
        raise ValueError("no turns to concatenate")
    bad = [c for c in turn_channels if not 0 <= c < num_channels]
    if bad:
        raise ValueError(f"turn channels {bad} out of range for {num_channels}")
    total = sum(int(w.shape[-1]) for w in turn_wavs)
    out = torch.zeros(num_channels, total, dtype=turn_wavs[0].dtype)
    offset = 0
    for wav, ch in zip(turn_wavs, turn_channels):
        flat = wav.reshape(-1)
        out[ch, offset : offset + flat.shape[0]] = flat
        offset += flat.shape[0]
    return out


def turn_spans(
    turn_wavs: Sequence[torch.Tensor],
    turn_channels: Sequence[int],
    turn_texts: Sequence[str],
    fs: int,
) -> list[dict[str, Any]]:
    """Turn spans in output time, the same shape the other modes emit."""
    spans, offset = [], 0
    for wav, ch, text in zip(turn_wavs, turn_channels, turn_texts):
        n = int(wav.reshape(-1).shape[0])
        spans.append(
            {
                "channel": int(ch),
                "text": text,
                "start": round(offset / fs, 6),
                "end": round((offset + n) / fs, 6),
            }
        )
        offset += n
    return spans


# --------------------------------------------------------------------------- #
# Sources: two ways to build BaselineItems, one generation core
# --------------------------------------------------------------------------- #
def _covomix2_items(cfg, training_config, fs: int) -> tuple[list[BaselineItem], dict]:
    """CoVoMix2: external prompts, PREDICTED turn durations."""
    testset = cfg.testset
    records = load_covomix2_testset(
        testset.root,
        testset.librispeech_root,
        OmegaConf.to_container(training_config, resolve=True)["dataset"][
            "preprocessor"
        ]["token_list"],
        num_channels=int(testset.get("num_channels", 2)),
    )
    dur_cfg = cfg.get("duration", {}) or {}
    duration_scale = float(dur_cfg.get("scale", DEFAULT_DURATION_SCALE))
    speed = float(dur_cfg.get("speed", 1.0))

    from egs3.conversational.tts.src.chunked_inference import estimate_turn_secs
    from egs3.conversational.tts.src.external_inference import _probe_duration_sec

    prompt_secs = [
        [_probe_duration_sec(p.audio_path) for p in r.prompts] for r in records
    ]
    predicted = [
        estimate_duration_sec(r, secs, duration_scale=duration_scale, speed=speed)
        for r, secs in zip(records, prompt_secs)
    ]
    indices, exclusions = select_records(records, predicted, cfg.selection)

    items = []
    for idx in indices:
        record = records[idx]
        secs = estimate_turn_secs(
            record, prompt_secs[idx], duration_scale=duration_scale, speed=speed
        )
        # `_load_prompt_wav` returns MONO (T,) - a CoVoMix2 prompt is one
        # LibriSpeech utterance, not a multichannel block - so index the
        # prompt LIST by channel, never the waveform.
        by_channel: dict[int, Any] = {}
        for p in record.prompts:
            by_channel[int(p.channel)] = p
        prompts = [by_channel[ch] for ch in range(record.num_channels)]
        items.append(
            BaselineItem(
                dialogue_id=record.dialogue_id,
                num_channels=record.num_channels,
                turn_channels=[t.channel for t in record.turns],
                turn_texts=[t.text for t in record.turns],
                turn_secs=list(secs),
                prompt_wavs=[_load_prompt_wav(p.audio_path, fs) for p in prompts],
                prompt_texts=[p.text for p in prompts],
                prompt_secs=[prompt_secs[idx][int(p.channel)] for p in prompts],
                extra_meta={
                    "testset": "covomix2-dialogue-testset",
                    "duration_policy": "predicted",
                    "duration_scale": duration_scale,
                    "speed": speed,
                },
            )
        )
    return items, exclusions


def _sssd_items(cfg, training_config, fs: int) -> tuple[list[BaselineItem], dict]:
    """SSSD: prompts cut from the session, REAL turn durations."""
    # Checked BEFORE the dataset is built: this is a config error, and
    # failing on it should not cost a manifest load first.
    manifest_path = cfg.selection.get("manifest")
    if not manifest_path:
        raise ValueError(
            "the SSSD baseline requires selection.manifest - the prompts must "
            "be the same pinned turns the system under test used, or the "
            "comparison is not controlled"
        )
    from egs3.conversational.tts.src.eval_manifest import load_eval_manifest

    _header, rows = load_eval_manifest(manifest_path)
    dataset = build_dataset(
        training_config,
        cfg.dataset.split,
        inference=True,
        manifest_path=cfg.dataset.get("manifest_path"),
        dataset_root=cfg.dataset.get("dataset_root"),
    )
    pools = _build_turn_pools(dataset.records)
    pinned = {r["window_id"]: r["prompts"] for r in rows}
    indices = _select_indices(dataset.records, cfg.selection, rows)

    items = []
    for idx in indices:
        record = dataset.records[idx]
        pool = pools.get(record.session_id, [])
        selected = _resolve_pinned_turns(pool, pinned[record.window_id], record)
        audio_path = dataset.dataset_root / record.audio_relpath
        prompt_wavs, prompt_texts, prompt_secs = [], [], []
        for ch, turn in enumerate(selected):
            block = read_audio_span(
                audio_path, record.sample_rate, turn.start, turn.end, fs
            )
            prompt_wavs.append(block[ch])
            prompt_texts.append(turn.text)
            prompt_secs.append(float(turn.end - turn.start))
        items.append(
            BaselineItem(
                dialogue_id=record.window_id,
                num_channels=record.num_channels,
                turn_channels=[t.channel for t in record.turns],
                turn_texts=[t.text for t in record.turns],
                turn_secs=[float(t.end - t.start) for t in record.turns],
                prompt_wavs=prompt_wavs,
                prompt_texts=prompt_texts,
                prompt_secs=prompt_secs,
                extra_meta={
                    "session_id": record.session_id,
                    "duration_policy": "reference",
                },
            )
        )
    return items, {"n_out_of_band": 0, "n_not_sampled": 0}


SOURCES = {"covomix2": _covomix2_items, "sssd": _sssd_items}


def run_concat_baseline(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
) -> dict[str, Any]:
    """Execute the concatenated-baseline infer stage; return counts."""
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

    source = str(cfg.get("source", "covomix2"))
    if source not in SOURCES:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(SOURCES)}"
        )
    items, exclusions = SOURCES[source](cfg, training_config, fs)

    shard_count = int(cfg.selection.get("shard_count", 1) or 1)
    shard_index = int(cfg.selection.get("shard_index", 0) or 0)
    # Shard by dialogue on predicted total cost so the shards finish together;
    # each dialogue is independent here, so no chain can straddle a shard.
    costs = [sum(it.turn_secs) for it in items]
    mine = assign_shard(list(range(len(items))), costs, shard_index, shard_count)
    logger.info(
        "concat baseline: %d/%d dialogues (source=%s, shard %d/%d, "
        "%d out of band, %d not sampled)",
        len(mine),
        len(items),
        source,
        shard_index,
        shard_count,
        exclusions.get("n_out_of_band", 0),
        exclusions.get("n_not_sampled", 0),
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

    token_list = OmegaConf.to_container(training_config, resolve=True)["dataset"][
        "preprocessor"
    ]["token_list"]
    token2id = make_token2id(read_vocab(token_list))

    test_dir = Path(cfg.inference_dir) / cfg.test_name
    for sub in ("meta", "wav", "prompt", "mix"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    meta_lines: list[str] = []
    wav_lines: list[str] = []
    prompt_lines: list[str] = []
    text_lines: list[str] = []
    mix_lines: list[str] = []
    samp = cfg.sampling
    n_turns_total = 0

    for i in tqdm([items[j] for j in mine], desc=f"infer[{MODE}]", unit="dialogue"):
        turn_wavs: list[torch.Tensor] = []
        elapsed_total = 0.0
        for ch, text, sec in zip(i.turn_channels, i.turn_texts, i.turn_secs):
            prompt = i.prompt_wavs[ch].reshape(1, -1)
            prompt_frames = int(prompt.shape[1]) // hop
            if prompt_frames < 1:
                raise ValueError(
                    f"{i.dialogue_id}: channel {ch} prompt is shorter than one "
                    f"hop ({prompt.shape[1]} samples)"
                )
            prompt_trimmed = prompt[:, : prompt_frames * hop]
            gen_frames = max(1, round(float(sec) * fs / hop))
            speech = torch.cat(
                [prompt_trimmed, torch.zeros(1, gen_frames * hop)], dim=1
            ).to(device)
            text_ids = plain_text_ids(i.prompt_texts[ch], text, token2id).to(device)
            wav, elapsed = generate_region(
                model,
                vocoder,
                speech,
                text_ids,
                prompt_frames,
                prompt_frames + gen_frames,
                steps=int(samp.steps),
                cfg_strength=float(samp.cfg_strength),
                sway_sampling_coef=float(samp.sway_sampling_coef),
                seed=samp.get("seed"),
            )
            turn_wavs.append(wav.reshape(-1).cpu())
            elapsed_total += float(elapsed)
            n_turns_total += 1

        laid = concat_timeline(turn_wavs, i.turn_channels, i.num_channels)
        wid = i.dialogue_id
        n = i.num_channels

        class _T:  # _reference_texts wants objects with .channel/.text
            def __init__(self, channel, text):
                self.channel, self.text = channel, text

        ref_texts = _reference_texts(
            [_T(c, t) for c, t in zip(i.turn_channels, i.turn_texts)], n
        )
        channels = []
        for ch in range(n):
            gen_rel = f"wav/{wid}_ch{ch}.wav"
            prompt_rel = f"prompt/{wid}_ch{ch}.wav"
            write_wav(test_dir / gen_rel, laid[ch], fs)
            write_wav(test_dir / prompt_rel, i.prompt_wavs[ch].reshape(-1), fs)
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
        write_wav(test_dir / mix_rel, laid.sum(dim=0) / n, fs)
        mix_lines.append(f"{wid} {mix_rel}")

        total_samples = int(laid.shape[1])
        meta = {
            "window_id": wid,
            "session_id": i.extra_meta.get("session_id", wid),
            "mode": MODE,
            "source": source,
            "sample_rate": fs,
            "num_channels": n,
            "window_duration_sec": round(total_samples / fs, 6),
            "has_reference_audio": source == "sssd",
            "turn_times": "concatenated",
            "rtf": (
                round(elapsed_total / (total_samples / fs), 6)
                if total_samples
                else None
            ),
            "baseline": {
                # Recorded so a reader of the metrics never has to infer why
                # this run's interaction numbers are degenerate.
                "engine": "stock F5 (counts=[1], zero-init gates)",
                "prompt_policy": "fixed per-speaker reference",
                "layout": "back-to-back, zero gap",
                "n_turns": len(turn_wavs),
                **i.extra_meta,
            },
            "mix_wav": mix_rel,
            "prompt": {
                "total_sec": round(sum(i.prompt_secs), 6),
                "turns": [
                    {
                        "channel": ch,
                        "text": i.prompt_texts[ch],
                        "duration_sec": round(i.prompt_secs[ch], 6),
                    }
                    for ch in range(n)
                ],
            },
            "turns": turn_spans(turn_wavs, i.turn_channels, i.turn_texts, fs),
            "channels": channels,
        }
        rel = f"meta/{wid}.json"
        (test_dir / rel).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_lines.append(f"{wid} {rel}")

    suffix = "" if shard_count == 1 else f".{shard_index}of{shard_count}"
    for name, lines in (
        ("meta", meta_lines),
        ("wav", wav_lines),
        ("prompt", prompt_lines),
        ("text", text_lines),
        ("mix", mix_lines),
    ):
        _write_scp(test_dir / f"{name}.scp{suffix}", lines)

    return {
        "n_selected": len(mine),
        "n_skipped": 0,
        "n_turns": n_turns_total,
        "n_other_shards": len(items) - len(mine),
    }
