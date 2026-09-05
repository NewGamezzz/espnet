"""Inference for the dual-prompt model: rebuild the training layout from audio.

``DualPromptInference`` reuses ``F5TTSInference`` from PR 6515 for model and
vocoder loading, sampling and vocoding, and replaces its tokenizer with the
recipe's phonemizer plus the shared layout helpers, so the text lane at
inference is built by the same code path the dataset uses in training.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torchaudio

from espnet3.systems.tts.f5_tts.inference import F5TTSInference
from src.layout import HOP, SR, TokenTable, build_text_ids
from src.layout import cond_frames as _cond_frames
from src.text.lemas_phonemizer import LEMASPhonemizer


class DualPromptInference(F5TTSInference):
    """Synthesize from a speaker prompt, an optional language prompt and text."""

    def __init__(
        self,
        train_config: str,
        checkpoint_path: str,
        token_list: str,
        lang_stats: str,
        device: str = "cpu",
        use_ema: bool = True,
        vocoder_name: str = "vocos",
        vocoder_path: Optional[str] = None,
        target_sample_rate: int = 24000,
        ode_solver_steps: int = 32,
        guidance_strength: float = 2.0,
        sway_sampling_coefficient: float = -1.0,
        speed: float = 1.0,
        target_rms: float = 0.1,
        lowpass_hz: Optional[float] = 8000.0,
        seed: Optional[int] = None,
    ):
        """Load model, token table, duration prior and vocoder.

        Args:
            train_config: Recipe training YAML (``model`` block).
            checkpoint_path: Lightning checkpoint.
            token_list: Token list written by ``create_token_list``.
            lang_stats: ``lang_stats.json`` from build (tokens per second).
            device: Torch device.
            use_ema: Prefer the EMA weights.
            vocoder_name / vocoder_path: As in ``F5TTSInference``.
            target_sample_rate: Output sample rate.
            ode_solver_steps / guidance_strength / sway_sampling_coefficient /
                speed / target_rms / seed: Sampling knobs, as the parent.
            lowpass_hz: Low-pass prompts at this cutoff (``None`` disables);
                training audio is 16 kHz-sourced, so full-band prompts are
                out of distribution (spec section 9).

        Example:
            .. code-block:: yaml

                model:
                  _target_: src.inference.DualPromptInference
                  train_config: ${recipe_dir}/conf/training_f5_base_dualprompt.yaml
                  checkpoint_path: ${exp_dir}/last.ckpt
                  token_list: ${data_dir}/tokens/tokens.txt
                  lang_stats: ${data_dir}/lang_stats.json
        """
        self.table = TokenTable(token_list)
        with open(lang_stats, encoding="utf-8") as f:
            self.lang_stats = json.load(f)
        self.lowpass_hz = lowpass_hz
        self.phonemizer = LEMASPhonemizer()
        super().__init__(
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            device=device,
            use_ema=use_ema,
            vocoder_name=vocoder_name,
            vocoder_path=vocoder_path,
            target_sample_rate=target_sample_rate,
            ode_solver_steps=ode_solver_steps,
            guidance_strength=guidance_strength,
            sway_sampling_coefficient=sway_sampling_coefficient,
            speed=speed,
            target_rms=target_rms,
            seed=seed,
        )

    def _build_tokenizer(self, config: dict) -> None:
        """The recipe owns tokenization (phonemizer + TokenTable)."""
        return None

    # ---- layout --------------------------------------------------------------
    def _prep_prompt(self, wav) -> Optional[torch.Tensor]:
        """Mono, optional low-pass, cut to the 768-sample prompt quantum."""
        if wav is None or len(wav) == 0:
            return None
        x = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
        if x.ndim > 1:
            x = x.mean(dim=-1)
        if self.lowpass_hz:
            x = torchaudio.functional.lowpass_biquad(x, SR, float(self.lowpass_hz))
        n = (len(x) // (3 * HOP)) * (3 * HOP)
        if n == 0:
            return None
        return x[:n].unsqueeze(0)

    def build_inputs(self, text, lang, spk_prompt_speech, lang_prompt_speech=None):
        """Build ``(cond_mel, text_ids, cond_frames, n_target_frames)``.

        Args:
            text: Target text.
            lang: Target language code.
            spk_prompt_speech: Speaker prompt waveform at 24 kHz.
            lang_prompt_speech: Language prompt waveform at 24 kHz, or ``None``.

        Returns:
            ``cond`` ``[1, cond_frames, D]`` (prompt mels only), ``ids``
            ``[1, T_text]``, ``cond_frames``, and the target frame count from
            the per-language tokens-per-second prior scaled by ``speed``.

        Example:
            >>> cond, ids, cf, n_tgt = infer.build_inputs("Hallo", "de", spk_wav)
        """
        spk = self._prep_prompt(spk_prompt_speech)
        lp = self._prep_prompt(lang_prompt_speech)
        parts, frames = [], []
        for w in (spk, lp):
            if w is None:
                frames.append(0)
                continue
            mel = self.cfm.mel_spec(w.to(self.device)).permute(0, 2, 1)
            mel = mel[:, : w.shape[-1] // HOP, :]
            parts.append(mel)
            frames.append(mel.shape[1])
        cf = _cond_frames(frames[0], frames[1])
        if parts:
            cond = torch.cat(parts, dim=1)
        else:
            cond = torch.zeros(1, 0, self.cfm.num_channels, device=self.device)
        phones = self.phonemizer.phonemize(text, lang)
        ids = torch.from_numpy(
            build_text_ids(frames[0], frames[1], lang, phones, self.table)
        ).unsqueeze(0)
        rate = float(self.lang_stats[lang]["tokens_per_sec"])
        n_tgt = int(round(len(phones) / rate * SR / HOP / self.speed))
        return cond, ids.to(self.device), cf, max(n_tgt, len(phones) + 1)

    @torch.no_grad()
    def __call__(self, text, lang, spk_prompt_speech, lang_prompt_speech=None, **_unused):
        """Synthesize one utterance.

        Args:
            text: Target text.
            lang: Target language code.
            spk_prompt_speech: Speaker prompt waveform (24 kHz).
            lang_prompt_speech: Language prompt waveform, optional.
            **_unused: Extra dataset keys, ignored.

        Returns:
            ``{"wav": float32 waveform}`` at ``target_sample_rate``.

        Example:
            >>> infer(text="Hallo Welt", lang="de", spk_prompt_speech=wav)["wav"].shape
            (52224,)
        """
        cond, ids, cf, n_tgt = self.build_inputs(text, lang, spk_prompt_speech, lang_prompt_speech)
        if cf == 0:  # no prompt at all: one silent frame keeps CFM.sample's shapes valid
            cond = torch.zeros(1, 1, self.cfm.num_channels, device=self.device)
        out, _ = self.cfm.sample(
            cond=cond,
            text=ids,
            duration=cf + n_tgt,
            lens=torch.tensor([cf], device=self.device),
            steps=self.ode_solver_steps,
            cfg_strength=self.guidance_strength,
            sway_sampling_coef=self.sway_sampling_coefficient,
            seed=self.seed,
        )
        mel = out[:, cf:, :].to(torch.float32)
        wav = self._vocode(mel.permute(0, 2, 1))
        return {"wav": np.asarray(wav, dtype=np.float32).reshape(-1)}


def build_output(data, model_output, idx):
    """Assemble the row the measure stage scores.

    Args:
        data: Dataset sample (``utt_id``, ``raw_text``, ``ref_wav_path``,
            ``lang_ref_wav_path``, ``gt_wav_path``).
        model_output: ``{"wav": ...}`` from :class:`DualPromptInference`.
        idx: Row index, used when ``utt_id`` is missing.

    Returns:
        ``{"utt_id", "text", "ref", "lang_ref", "gt", "wav"}``; ``ref`` is the
        speaker prompt (SIM), ``lang_ref`` the language prompt (leak probe).

    Example:
        >>> sorted(build_output(sample, {"wav": wav}, 0))
        ['gt', 'lang_ref', 'ref', 'text', 'utt_id', 'wav']
    """
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("TTS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "text": str(data.get("raw_text", "")),
        "ref": str(data.get("ref_wav_path", "")),
        "lang_ref": str(data.get("lang_ref_wav_path", "")),
        "gt": str(data.get("gt_wav_path", "")),
        "wav": np.asarray(wav, dtype=np.float32).reshape(-1),
    }
