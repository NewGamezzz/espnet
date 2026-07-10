#!/usr/bin/env python3
"""Sanity generation from one dev window (the go/no-go listening artifact).

Takes one window from the step-2 manifests, keeps the first ``--prompt-sec``
of every channel as the acoustic prompt (the infilling mask covers the
remainder), conditions each branch on its full masked script, runs the
multi-branch ODE inference with the exchanges active and CFG as in the
existing F5 inference path, vocodes each channel with Vocos, and writes
per-channel wavs plus a mixdown next to a text dump of the masked scripts.

Separated channels with sensible turn-taking = POC signal; audio quality is
NOT a criterion here.  The dev-window prompt is the degenerate case of the
staggered dialogue-history voice conditioning (PLAN step 3): external-voice
demos would place reference clips as opening turns instead.

Example (from the recipe dir, after training):
    python local/generate_dev.py \
        --training_config conf/training_poc.yaml \
        --ckpt exp/train_poc_multibranch_f5/checkpoints/last.ckpt \
        --index 0 --prompt_sec 3.0 --out_dir exp/generate_dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_config", type=Path, default=Path("conf/training_poc.yaml")
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Lightning checkpoint; omit to hear the assembled "
        "pretrained model (zero-init gates = independent F5 passes).",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Load raw weights even when the ckpt has EMA weights.",
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--window_id", default=None, help="Select by window id instead of --index."
    )
    parser.add_argument("--prompt_sec", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", type=Path, default=Path("exp/generate_dev"))
    return parser.parse_args()


def load_model(config, ckpt_path: Path | None, use_ema: bool, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Resolve while the node is still attached to the root config (the model
    # block interpolates ${data_dir}/${pretrained_dir}/${seed}).
    model_config = OmegaConf.to_container(config.model, resolve=True)
    if ckpt_path is not None:
        # The training ckpt supersedes every pretrained weight: skip the
        # native checkpoint load, the embedding surgery, and the provenance
        # check (and with them the hard dependency on downloads/ existing).
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
            print(f"loaded EMA weights from {ckpt_path}")
        else:
            state = ckpt.get("state_dict", ckpt)
            print(f"loaded raw weights from {ckpt_path}")
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_vocoder(device: torch.device):
    # Same Vocos as espnet2/tts/f5/inference.py::F5TTSInference._load_vocoder.
    from vocos import Vocos

    return Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()


def main() -> None:
    args = parse_args()
    from omegaconf import OmegaConf

    from egs3.conversational.tts.dataset.dataset import ConversationDataset
    from egs3.conversational.tts.dataset.preprocessing.text import render_tokens
    from egs3.conversational.tts.dataset.preprocessor import (
        ConversationalTextPreprocessor,
    )

    config = OmegaConf.load(args.training_config)
    device = torch.device(args.device)

    dataset = ConversationDataset(
        split=args.split,
        recipe_dir=config.recipe_dir,
        fs=int(config.sample_rate),
        permute_channels=False,
        inference=True,
    )
    if args.window_id is not None:
        matches = [
            i for i, r in enumerate(dataset.records) if r.window_id == args.window_id
        ]
        if not matches:
            raise SystemExit(f"window id {args.window_id!r} not in {args.split}")
        index = matches[0]
    else:
        index = args.index
    sample = dataset[index]
    preprocessor = ConversationalTextPreprocessor(
        token_list=OmegaConf.to_container(config, resolve=True)["dataset"][
            "preprocessor"
        ]["token_list"]
    )
    sample = preprocessor(str(index), sample)

    n = sample["num_channels"]
    speech = sample["speech"].to(device)  # (N, T_wav) at 24 kHz
    fs = dataset.fs
    hop = int(config.hop_length)
    prompt_samples = round(args.prompt_sec * fs)
    if prompt_samples >= speech.shape[1]:
        raise SystemExit(
            f"--prompt_sec {args.prompt_sec} covers the whole "
            f"{speech.shape[1] / fs:.1f} s window"
        )
    prompt_frames = prompt_samples // hop
    total_frames = speech.shape[1] // hop

    text = torch.nn.utils.rnn.pad_sequence(
        sample["text"], batch_first=True, padding_value=-1
    ).to(device)

    model = load_model(config, args.ckpt, use_ema=not args.no_ema, device=device)
    vocoder = load_vocoder(device)

    # Every channel keeps its first prompt_frames as the acoustic prompt
    # (lens), the ODE fills the remainder jointly with the exchanges active.
    lens = torch.full((n,), prompt_frames, device=device, dtype=torch.long)
    with torch.inference_mode():
        mel, _ = model.cfm.sample(
            cond=speech,  # raw wave rows; CFM extracts the mel itself
            text=text,
            duration=total_frames,
            counts=[n],
            lens=lens,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            seed=args.seed,
        )
        gen_mel = mel[:, prompt_frames:, :].to(torch.float32)  # drop the prompt
        wavs = vocoder.decode(gen_mel.permute(0, 2, 1)).cpu()  # (N, T_gen)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_id = sample["window_id"]
    try:
        import soundfile as sf

        def write(path, wav):
            sf.write(str(path), wav.numpy(), fs)

    except ImportError:
        import torchaudio

        def write(path, wav):
            torchaudio.save(str(path), wav.unsqueeze(0), fs)

    for ch in range(n):
        write(args.out_dir / f"{window_id}_ch{ch}.wav", wavs[ch])
    write(args.out_dir / f"{window_id}_mix.wav", wavs.sum(dim=0) / n)

    lines = [
        f"window {window_id} ({args.split}[{index}])",
        f"prompt: first {args.prompt_sec} s of every channel",
        "",
    ]
    for ch, tokens in enumerate(_branch_tokens(sample)):
        lines.append(f"[ch{ch}] {render_tokens(tokens)}")
    (args.out_dir / f"{window_id}_scripts.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {n} channel wavs + mixdown + scripts to {args.out_dir}")


def _branch_tokens(sample):
    from egs3.conversational.tts.dataset.preprocessing.text import build_branch_texts

    return build_branch_texts(sample["turns"], sample["num_channels"])


if __name__ == "__main__":
    main()
