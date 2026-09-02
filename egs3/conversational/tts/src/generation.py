"""Shared inference primitives for the conversational multi-branch F5 recipe.

Factored out of ``local/generate_dev.py`` so both the quick listening tool and
the espnet3 ``infer`` stage (``src/inference.py``) load the model / vocoder /
dataset the same way and run identical conditioning + sampling.

Model, vocoder and dataset are passed in as objects (or built through the
loaders here), so tests drive the stage on CPU with a tiny random-init DiT and
a fake vocoder while the real run uses the assembled F5 + Vocos.

Nothing in this module - or in ``src/inference.py`` - is imported by the
training path.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import torchaudio

from espnet2.fileio.sound_scp import soundfile_read


def load_model(config, ckpt_path: Path | None, use_ema: bool, device: torch.device):
    """Assemble the multi-branch F5 model and (optionally) load a Lightning ckpt.

    Verbatim behaviour of the former ``generate_dev.load_model``: an omitted
    ``ckpt_path`` keeps the assembled pretrained model (zero-init gates); a
    training ckpt supersedes every pretrained weight (native load / embedding
    surgery / provenance check skipped).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    model_config = OmegaConf.to_container(config.model, resolve=True)
    if ckpt_path is not None:
        model_config["pretrained_ckpt"] = None
    model = instantiate(model_config)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if use_ema and "ema_model_state_dict" in ckpt:
            prefix = "ema_model."
            state = {
                k[len(prefix) :]: v
                for k, v in ckpt["ema_model_state_dict"].items()
                if k.startswith(prefix)
            }
        else:
            state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_vocoder(device: torch.device):
    """The same Vocos as ``espnet2/tts/f5/inference.py``."""
    from vocos import Vocos

    return Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()


def build_dataset(
    config,
    split: str,
    *,
    inference: bool = True,
    manifest_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    min_active_speakers: int = 1,
):
    """Build a ``ConversationDataset`` (channel permutation off for inference)."""
    from egs3.conversational.tts.dataset.dataset import ConversationDataset

    return ConversationDataset(
        split=split,
        recipe_dir=config.recipe_dir,
        fs=int(config.sample_rate),
        permute_channels=False,
        inference=inference,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        min_active_speakers=min_active_speakers,
    )


def read_audio_span(
    audio_path: str | Path,
    source_sample_rate: int,
    t0: float,
    t1: float,
    target_fs: int,
    channels: Sequence[int] | None = None,
) -> torch.Tensor:
    """Seek-read a multichannel ``[t0, t1)`` span of a session file and resample.

    Mirrors ``ConversationDataset._load_speech``'s read/resample path (seek in
    source-rate samples, ``soundfile_read`` with ``always_2d=True``, resample
    only if the target rate differs) so prompt-turn audio is processed
    identically to window audio.  Channel-count validation is NOT repeated
    here: the caller reads the same session file through the dataset loader
    first, which already raises on a manifest mismatch.  ``channels`` selects
    source columns (in the given order) before resampling - the window
    record's ``row_channels`` - and None keeps every column.  Returns
    ``(N, T)`` at ``target_fs``.
    """
    start = round(t0 * source_sample_rate)
    stop = round(t1 * source_sample_rate)
    array, rate = soundfile_read(
        str(audio_path), dtype="float32", start=start, end=stop, always_2d=True
    )
    if rate != source_sample_rate:
        raise RuntimeError(
            f"{audio_path}: sample rate {rate} != expected {source_sample_rate}"
        )
    if channels is not None:
        if max(channels) >= array.shape[1]:
            raise RuntimeError(
                f"{audio_path}: channels {tuple(channels)} but the file has "
                f"{array.shape[1]} columns"
            )
        array = array[:, list(channels)]
    speech = torch.from_numpy(array.T.copy())  # (N, T_src)
    if target_fs != source_sample_rate:
        speech = torchaudio.functional.resample(
            speech, orig_freq=source_sample_rate, new_freq=target_fs
        )
    return speech


def build_preprocessor(config):
    """Build the recipe text preprocessor from the training config token list."""
    from omegaconf import OmegaConf

    from egs3.conversational.tts.dataset.preprocessor import (
        ConversationalTextPreprocessor,
    )

    token_list = OmegaConf.to_container(config, resolve=True)["dataset"][
        "preprocessor"
    ]["token_list"]
    return ConversationalTextPreprocessor(token_list=token_list)


def pad_branch_text(sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Pack a preprocessed sample's per-branch token lists into a padded tensor."""
    return torch.nn.utils.rnn.pad_sequence(
        sample["text"], batch_first=True, padding_value=-1
    ).to(device)


def generate_region(
    model,
    vocoder,
    speech: torch.Tensor,
    text: torch.Tensor,
    prompt_frames: int,
    total_frames: int,
    *,
    steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    seed: int | None,
) -> tuple[torch.Tensor, float]:
    """Run the multi-branch ODE and vocode the infilled region.

    ``speech`` are the raw per-channel prompt+target waves ``(N, T_wav)``; the
    first ``prompt_frames`` of every channel act as the acoustic prompt and the
    ODE fills the remainder jointly (exchanges active, CFG as in F5).  Returns
    the generated-region waves ``(N, T_gen)`` and the wall-clock seconds spent.
    """
    n = speech.shape[0]
    device = speech.device
    lens = torch.full((n,), prompt_frames, device=device, dtype=torch.long)
    start = time.perf_counter()
    with torch.inference_mode():
        mel, _ = model.cfm.sample(
            cond=speech,
            text=text,
            duration=total_frames,
            counts=[n],
            lens=lens,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
        )
        gen_mel = mel[:, prompt_frames:, :].to(torch.float32)  # drop the prompt
        wavs = vocoder.decode(gen_mel.permute(0, 2, 1)).cpu()  # (N, T_gen)
    elapsed = time.perf_counter() - start
    return wavs, elapsed


@dataclass
class GenerationItem:
    """One dialogue's inputs to a batched ODE call.

    ``speech`` is the raw per-channel conditioning wave ``(N, T_wav)`` -
    prompt followed by a zero generated region - and ``text`` the padded
    per-channel token ids ``(N, T_text)`` (pad -1).  ``prompt_frames`` /
    ``total_frames`` delimit the infilled region exactly as in
    :func:`generate_region`.
    """

    speech: torch.Tensor
    text: torch.Tensor
    prompt_frames: int
    total_frames: int
    # Optional per-channel classifier-free guidance, one value per row of
    # ``speech``; ``None`` = the batch-wide ``cfg_strength`` (bit-identical
    # to the scalar path).
    cfg_per_channel: tuple[float, ...] | None = None


def _autocast(device: torch.device, dtype: str | None):
    """Autocast context for the ODE, or a no-op when ``dtype`` is ``None``.

    The vocoder is deliberately kept OUTSIDE this context by the callers:
    reduced precision is validated for the transformer (the rotary embedding
    carries explicit autocast guards) but not for Vocos' ISTFT head.
    """
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def generate_batch(
    model,
    vocoder,
    items: list[GenerationItem],
    *,
    steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    seed: int | None,
    autocast_dtype: str | None = None,
) -> tuple[list[torch.Tensor], float]:
    """Run ONE multi-branch ODE over several dialogues packed row-wise.

    The packed layout is the training-time one: all dialogues' channels are
    row-stacked, ``counts`` groups rows into conversations, and the injected
    exchanges mix strictly within a conversation (``conv_id`` blocks), so
    dialogues cannot contaminate each other.  Per-row ``lens`` carry each
    dialogue's prompt length and per-row ``duration`` its total frames; rows
    beyond a dialogue's duration are padding whose output is discarded.
    Padded rows attend exactly as padded training batches did
    (``attn_mask_enabled`` false), so batching is in-distribution - callers
    keep padding waste low by batching length-sorted dialogues.

    ``seed`` seeds the RNG ONCE for the whole batch (the per-batch analogue
    of ``generate_region``'s per-call seeding): a batch of one is
    bit-identical to the sequential path, while noise inside a larger batch
    depends on batch composition - which the caller keeps a pure function of
    config, never of shard membership.

    Returns one ``(N, T_gen)`` generated-region wave per item (vocoded per
    dialogue at its exact length, outside autocast) and the wall-clock
    seconds spent on the whole batch.
    """
    if not items:
        return [], 0.0
    device = items[0].speech.device
    counts = [item.speech.shape[0] for item in items]
    max_samples = max(item.speech.shape[1] for item in items)
    max_tokens = max(item.text.shape[1] for item in items)
    cond = torch.cat(
        [F.pad(item.speech, (0, max_samples - item.speech.shape[1])) for item in items]
    )
    text = torch.cat(
        [
            F.pad(item.text, (0, max_tokens - item.text.shape[1]), value=-1)
            for item in items
        ]
    )
    lens = torch.cat(
        [
            torch.full((n,), item.prompt_frames, device=device, dtype=torch.long)
            for item, n in zip(items, counts)
        ]
    )
    duration = torch.cat(
        [
            torch.full((n,), item.total_frames, device=device, dtype=torch.long)
            for item, n in zip(items, counts)
        ]
    )
    # Per-row guidance only when some item asks for it; otherwise the scalar
    # goes through untouched, which keeps every existing run bit-identical.
    cfg: float | torch.Tensor = float(cfg_strength)
    if any(item.cfg_per_channel is not None for item in items):
        rows: list[float] = []
        for item, n in zip(items, counts):
            per = item.cfg_per_channel
            if per is None:
                rows.extend([float(cfg_strength)] * n)
                continue
            if len(per) != n:
                raise ValueError(
                    f"cfg_per_channel has {len(per)} values for {n} channels"
                )
            rows.extend(float(v) for v in per)
        cfg = torch.tensor(rows, device=device, dtype=torch.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        with _autocast(device, autocast_dtype):
            mel, _ = model.cfm.sample(
                cond=cond,
                text=text,
                duration=duration,
                counts=counts,
                lens=lens,
                steps=steps,
                cfg_strength=cfg,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        mel = mel.to(torch.float32)
        wavs = []
        offset = 0
        for item, n in zip(items, counts):
            gen_mel = mel[
                offset : offset + n, item.prompt_frames : item.total_frames, :
            ]
            wavs.append(vocoder.decode(gen_mel.permute(0, 2, 1)).cpu())
            offset += n
    elapsed = time.perf_counter() - start
    return wavs, elapsed


def resynth_region(model, vocoder, region_wav: torch.Tensor) -> torch.Tensor:
    """Round-trip a raw wave through the F5 mel front-end + vocoder.

    ``region_wav`` is ``(N, T)``; the model's ``MelSpec`` yields ``(N, n_mel,
    T_frames)`` (already in the vocoder's channels-first orientation, no
    permute), which is decoded back to ``(N, T_out)``.
    """
    with torch.inference_mode():
        mel = model.cfm.mel_spec(region_wav)  # (N, n_mel, T_frames)
        wavs = vocoder.decode(mel).cpu()
    return wavs


def write_wav(path: Path, wav: torch.Tensor, fs: int) -> None:
    """Write a 1-D wave tensor, preferring soundfile with a torchaudio fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), wav.detach().cpu().numpy(), fs)
    except ImportError:  # pragma: no cover - soundfile is a hard dep in practice
        import torchaudio

        torchaudio.save(str(path), wav.detach().cpu().unsqueeze(0), fs)
