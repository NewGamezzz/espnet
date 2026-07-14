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

import time
from pathlib import Path
from typing import Any

import torch


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
