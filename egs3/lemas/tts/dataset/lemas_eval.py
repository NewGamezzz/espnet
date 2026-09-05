"""LEMAS-eval dataset for the infer stage (prompts pinned by the manifest).

Rows come from ``local/prepare_lemas_eval.py``. Output keys feed
``src.inference.DualPromptInference`` through ``input_key`` and
``src.inference.build_output`` for scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


class LEMASEvalDataset(torch.utils.data.Dataset):
    """One LEMAS-eval row per item, optionally filtered to one language."""

    def __init__(
        self,
        manifest_path,
        use_lang_prompt: bool = True,
        fs: int = 24000,
        lang: Optional[str] = None,
    ):
        """Read the manifest.

        Args:
            manifest_path: Output of ``build_eval_manifest``.
            use_lang_prompt: Emit ``lang_prompt_speech`` (arm A) or not (arm B).
            fs: Sample rate the clips are resampled to.
            lang: Keep only rows of this language when given.

        Raises:
            RuntimeError: If no row matches.

        Example:
            .. code-block:: yaml

                test:
                  - name: lemas_eval_de
                    data_src: egs3.lemas.tts.dataset.lemas_eval
                    data_src_args: {manifest_path: ..., lang: de}
        """
        self.use_lang_prompt, self.fs = use_lang_prompt, fs
        self.rows = []
        with Path(manifest_path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = line.rstrip("\n").split("\t")
                    if lang is None or r[1] == lang:
                        self.rows.append(r)
        if not self.rows:
            raise RuntimeError(f"Empty eval manifest: {manifest_path} (lang={lang})")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.fs:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, self.fs).numpy()
        return np.asarray(wav, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        utt, lang, text, spk, lp, gt = self.rows[idx]
        sample = {
            "utt_id": utt,
            "text": text,
            "lang": lang,
            "raw_text": text,
            "spk_prompt_speech": self._load(spk),
            "ref_wav_path": spk,
            "lang_ref_wav_path": lp,
            "gt_wav_path": gt,
        }
        if self.use_lang_prompt:
            sample["lang_prompt_speech"] = self._load(lp)
        return sample


Dataset = LEMASEvalDataset
