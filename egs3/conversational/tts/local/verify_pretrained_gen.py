#!/usr/bin/env python3
"""Zero-gate listening artifacts from the assembled pretrained model.

Unlike ``local/generate_dev.py`` this needs NO SSSD manifests: it builds the
windows inline from a reference clip, so the assembled model (pretrained
F5TTS_Base + surgery + zero-gate TAC injection) can be heard before any
training data or checkpoint exists.

Artifacts written to ``--out_dir``:

* ``single_multibranch.wav``  - single-channel continuation of the reference
  clip through the assembled model with ``counts=[1]``.  With zero gates this
  must sound exactly like stock F5: correct words in the reference voice.
* ``single_baseline.wav``     - the same inputs through the baseline espnet2
  ``CFM`` (A/B check; the max mel difference is printed and should be ~0,
  matching tests/test_pretrained_real.py's parity test).
* ``twochannel_ch{0,1}.wav`` + ``twochannel_mix.wav`` + ``_scripts.txt`` -
  a synthetic two-speaker window with masked ``<turn>``/``<OTHER>`` scripts
  (``--two_channel``).  Qualitative only: the new token embeddings are
  warm-started but untrained, so this is the documented pre-finetuning
  baseline, not a pass/fail gate.

Example (from the recipe dir):
    python local/verify_pretrained_gen.py --steps 32 --two_channel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RECIPE_DIR = Path(__file__).resolve().parents[1]
TARGET_RMS = 0.1  # F5's inference reference loudness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=RECIPE_DIR / "conf" / "training_poc.yaml"
    )
    parser.add_argument(
        "--ref_wav", type=Path, default=RECIPE_DIR / "downloads/ref/basic_ref_en.wav"
    )
    parser.add_argument(
        "--ref_text",
        default="Some call me nature, others call me mother nature.",
    )
    parser.add_argument(
        "--gen_text",
        default="Verification is the art of asking a model to prove it still "
        "remembers what it was taught.",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--two_channel", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=RECIPE_DIR / "exp" / "verify")
    return parser.parse_args()


def build_models(config_path: Path, skip_baseline: bool):
    """The assembled multi-branch model (+ optionally the baseline CFM)."""
    from omegaconf import OmegaConf

    from egs3.conversational.tts.dataset.preprocessing.text import extend_vocab
    from egs3.conversational.tts.dataset.preprocessor import read_vocab
    from egs3.conversational.tts.src.build_model import build_multibranch_f5
    from espnet2.tts.f5.backbones.dit import DiT
    from espnet2.tts.f5.cfm import CFM
    from espnet2.tts.f5.inference import F5TTSInference

    config = OmegaConf.load(config_path)
    config.recipe_dir = str(RECIPE_DIR)
    model_config = OmegaConf.to_container(config, resolve=True)["model"]

    base_vocab = Path(model_config["pretrained_vocab"])
    base = read_vocab(base_vocab)
    tokens_dir = Path(tempfile.mkdtemp(prefix="verify_tokens_"))
    vocab_file = tokens_dir / "vocab.txt"
    vocab_file.write_text("\n".join(extend_vocab(base)) + "\n", encoding="utf-8")
    data = base_vocab.read_bytes()
    meta_file = tokens_dir / "vocab_meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "base_vocab_sha256": hashlib.sha256(data).hexdigest(),
                "base_vocab_size": len(data.decode("utf-8").splitlines()),
            }
        ),
        encoding="utf-8",
    )

    assembled = build_multibranch_f5(
        vocab_file=str(vocab_file),
        arch=model_config["arch"],
        cfm=model_config["cfm"],
        exchange=model_config["exchange"],
        feats_extract=model_config["feats_extract"],
        pretrained_ckpt=model_config["pretrained_ckpt"],
        pretrained_vocab=str(base_vocab),
        vocab_meta=str(meta_file),
        init_noise_scale=model_config["init_noise_scale"],
        init_seed=0,
    ).eval()

    baseline = None
    if not skip_baseline:
        arch, fe = model_config["arch"], model_config["feats_extract"]
        baseline = CFM(
            DiT(mel_dim=fe["n_mels"], text_num_embeds=len(base), **arch),
            num_channels=fe["n_mels"],
            odeint_kwargs=dict(method="euler"),
            mel_spec_kwargs=dict(
                n_fft=fe["n_fft"],
                hop_length=fe["hop_length"],
                win_length=fe["win_length"],
                n_mel_channels=fe["n_mels"],
                target_sample_rate=fe["fs"],
                mel_spec_type=fe["mel_spec_type"],
            ),
        )
        raw = F5TTSInference._load_native_f5_state(
            model_config["pretrained_ckpt"], use_ema=True
        )
        baseline_keys = set(baseline.state_dict())
        baseline.load_state_dict(
            {
                k: v
                for k, v in raw.items()
                if not (k.startswith("mel_spec.") and k not in baseline_keys)
            },
            strict=True,
        )
        baseline.eval()

    hop = int(model_config["feats_extract"]["hop_length"])
    fs = int(model_config["feats_extract"]["fs"])
    return assembled, baseline, base, extend_vocab(base), hop, fs


def encode(text: str, tokens: list[str]) -> torch.Tensor:
    from egs3.conversational.tts.dataset.preprocessing.text import (
        make_token2id,
        normalize_text,
        vocab_charset,
    )

    token2id = make_token2id(tokens)
    normalized = normalize_text(text, vocab_charset(tokens))
    return torch.tensor([[token2id[c] for c in normalized]], dtype=torch.long)


def load_reference(path: Path, fs: int) -> torch.Tensor:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    if sr != fs:
        import torchaudio

        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, fs).squeeze(0)
    return wav


def rms_normalize(wav: torch.Tensor) -> tuple[torch.Tensor, float]:
    rms = float(torch.sqrt(torch.mean(torch.square(wav))))
    if rms < TARGET_RMS:
        wav = wav * TARGET_RMS / rms
    return wav, rms


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    from vocos import Vocos

    assembled, baseline, base_tokens, ext_tokens, hop, fs = build_models(
        args.config, args.skip_baseline
    )
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

    ref_wav = load_reference(args.ref_wav, fs)
    ref_norm, ref_rms = rms_normalize(ref_wav)
    ref_frames = ref_norm.shape[0] // hop

    def vocode(mel: torch.Tensor) -> torch.Tensor:
        wav = vocoder.decode(mel.to(torch.float32).permute(0, 2, 1)).cpu()
        if ref_rms < TARGET_RMS:
            wav = wav * ref_rms / TARGET_RMS
        return wav

    sample_kwargs = dict(
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        seed=args.seed,
    )

    # ---- single channel: continuation of the reference clip -------------
    # Same recipe as F5TTSInference.infer_one: text = ref + gen transcript,
    # duration extends the reference proportionally to the gen text length.
    ids = encode(args.ref_text + " " + args.gen_text, ext_tokens)
    gen_frames = int(
        ref_frames * len(args.gen_text.encode()) / len(args.ref_text.encode())
    )
    duration = ref_frames + gen_frames
    cond = ref_norm.unsqueeze(0)  # raw wave; CFM extracts the mel itself

    print(f"single-channel: {duration} frames total, {args.steps} steps ...")
    with torch.inference_mode():
        mel_multi, _ = assembled.cfm.sample(
            cond, ids, duration, counts=[1], **sample_kwargs
        )
    wav = vocode(mel_multi[:, ref_frames:, :])
    sf.write(str(args.out_dir / "single_multibranch.wav"), wav[0].numpy(), fs)

    if baseline is not None:
        ids_base = encode(args.ref_text + " " + args.gen_text, base_tokens)
        with torch.inference_mode():
            mel_base, _ = baseline.sample(cond, ids_base, duration, **sample_kwargs)
        wav_base = vocode(mel_base[:, ref_frames:, :])
        sf.write(str(args.out_dir / "single_baseline.wav"), wav_base[0].numpy(), fs)
        diff = float((mel_multi - mel_base).abs().max())
        print(f"A/B max |mel diff| multibranch vs baseline: {diff:.2e} (expect ~0)")

    # ---- two channels: synthetic masked-script window --------------------
    if args.two_channel:
        from egs3.conversational.tts.dataset.preprocessing.text import (
            build_branch_texts,
            encode_tokens,
            make_token2id,
            normalize_text,
            render_tokens,
            vocab_charset,
        )

        charset = vocab_charset(ext_tokens)
        token2id = make_token2id(ext_tokens)
        turns = [
            SimpleNamespace(channel=0, text=normalize_text(args.ref_text, charset)),
            SimpleNamespace(
                channel=1,
                text=normalize_text(
                    "that is very interesting, tell me more about it.", charset
                ),
            ),
            SimpleNamespace(
                channel=0,
                text=normalize_text("well, let me explain how it works.", charset),
            ),
        ]
        branches = build_branch_texts(turns, 2)
        text = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(encode_tokens(tokens, token2id), dtype=torch.long)
                for tokens in branches
            ],
            batch_first=True,
            padding_value=-1,
        )

        # Channel 0's prompt is the reference clip; channel 1 is silent while
        # channel 0 speaks (the dev-window degenerate case of dialogue-history
        # conditioning).  The ODE fills the remainder of both rows jointly.
        reply_frames = int(
            ref_frames
            * sum(len(t.text.encode()) for t in turns[1:])
            / max(len(turns[0].text.encode()), 1)
        )
        total = ref_frames + reply_frames
        cond2 = torch.zeros(2, total * hop)
        cond2[0, : ref_norm.shape[0]] = ref_norm
        lens = torch.full((2,), ref_frames, dtype=torch.long)

        print(f"two-channel: {total} frames total, {args.steps} steps ...")
        with torch.inference_mode():
            mel2, _ = assembled.cfm.sample(
                cond2, text, total, counts=[2], lens=lens, **sample_kwargs
            )
        wavs = vocode(mel2[:, ref_frames:, :])
        for ch in range(2):
            sf.write(str(args.out_dir / f"twochannel_ch{ch}.wav"), wavs[ch].numpy(), fs)
        sf.write(
            str(args.out_dir / "twochannel_mix.wav"),
            (wavs.sum(dim=0) / 2).numpy(),
            fs,
        )
        lines = ["synthetic two-channel window (zero gates, untrained new tokens)", ""]
        for ch, tokens in enumerate(branches):
            lines.append(f"[ch{ch}] {render_tokens(tokens)}")
        (args.out_dir / "twochannel_scripts.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    print(f"artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
