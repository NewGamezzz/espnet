"""Seed-TTS eval dataset: target text plus a fixed prompt per utterance.

Unlike the LibriTTS dataset there is no reference *selection* here. Seed-TTS
ships the prompt with each row, so ref_speech and ref_text come straight from
the manifest and the eval is exactly the published protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset


class SeedTTSDataset(TorchDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        fs: int | None = None,
        inference: bool = True,
    ) -> None:
        self.fs = fs
        self.inference = inference
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self._rows = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._rows.append(tuple(line.rstrip("\n").split("\t", 4)))
        if not self._rows:
            raise RuntimeError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def _read(self, path: str) -> np.ndarray:
        audio, audio_fs = sf.read(path)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.fs is not None and audio_fs != self.fs:
            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio),
                    orig_freq=audio_fs,
                    new_freq=self.fs,
                )
                .numpy()
                .astype(np.float32)
            )
        return audio

    def __getitem__(self, idx: int) -> dict[str, Any]:
        utt_id, wav_path, text, prompt_wav, prompt_text = self._rows[int(idx)]
        sample: dict[str, Any] = {
            "text": text,
            "ref_speech": self._read(prompt_wav),
            "ref_text": prompt_text,
        }
        if self.inference:
            sample.update(
                {
                    "utt_id": np.asarray(utt_id),
                    "wav_path": wav_path,
                    "raw_text": text,
                    "ref_wav_path": prompt_wav,
                }
            )
        return sample


# DataOrganizer resolves a dataset entry via getattr(module, "Dataset"), so
# expose the class under that name. This is what lets an inference config
# select this dataset with `data_src: egs3.emilia.tts.dataset.seedtts`
# instead of the recipe's default Dataset (EmiliaDataset).
Dataset = SeedTTSDataset
