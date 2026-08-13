"""Emilia dataset backed by columnar manifests.

At 37M rows a list of per-row Python objects costs tens of GB per DataLoader
worker and defeats fork copy-on-write, because refcount updates dirty the
pages. Every column here is a numpy array or a single bytes buffer, so the
pages stay shared.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

_UTT_ID_WIDTH = 32


def _load_cfg(recipe_dir: Path) -> tuple[dict, dict]:
    cfg_path = Path(recipe_dir) / "dataset" / "config.yaml"
    if not cfg_path.is_file():
        config_resource = resources.files(__package__).joinpath("config.yaml")
        with resources.as_file(config_resource) as fallback:
            cfg_path = fallback
    cfg = load_config_with_defaults(str(cfg_path), resolve=False)
    return cfg["builder"], cfg["dataset"]


class EmiliaDataset(TorchDataset):
    """Emilia utterances as ``{"text": str, "speech": np.ndarray}``.

    Args:
        split: ``"train"`` or ``"valid"``.
        recipe_dir: Recipe root; defaults to this file's grandparent.
        manifest_path: Overrides the split-keyed default.
        load_speech: When False, ``speech`` is omitted (used by create_shape).
        fs: Target sample rate; resamples when the file differs.
        inference: Adds ``utt_id``, ``wav_path`` and ``raw_text``.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        fs: int | None = None,
        inference: bool = False,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        self.fs = fs
        self.inference = inference

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        builder_cfg, dataset_cfg = _load_cfg(recipe_root)
        self.corpus_root = Path(builder_cfg["corpus_root"]) / "emilia"
        self.audio_suffix = builder_cfg.get("audio_suffix", ".mp3")
        data_dir = recipe_root / builder_cfg["data_path"]

        if manifest_path is not None:
            resolved = Path(manifest_path)
            if not resolved.is_absolute():
                resolved = (recipe_root / resolved).resolve()
        else:
            split_paths = dataset_cfg["split_manifest_paths"]
            if split not in split_paths:
                raise FileNotFoundError(
                    f"Unknown split {split!r}; expected one of "
                    f"{sorted(split_paths)}"
                )
            resolved = data_dir / split_paths[split]
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifest not found: {resolved}")
        self.manifest_path = resolved

        shard_table_path = data_dir / builder_cfg["shard_table_path"]
        self._shards = shard_table_path.read_text(
            encoding="utf-8"
        ).splitlines()

        self._load_columns(resolved)

    def _load_columns(self, path: Path) -> None:
        utt_ids: list[bytes] = []
        shard_idx: list[int] = []
        durations: list[float] = []
        text_chunks: list[bytes] = []
        offsets: list[int] = [0]
        cursor = 0

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                utt_id, shard, _lang, duration, text = line.rstrip(
                    "\n"
                ).split("\t", 4)
                utt_ids.append(utt_id.encode("ascii"))
                shard_idx.append(int(shard))
                durations.append(float(duration))
                blob = text.encode("utf-8")
                text_chunks.append(blob)
                cursor += len(blob)
                offsets.append(cursor)

        if not utt_ids:
            raise RuntimeError(f"Manifest is empty: {path}")

        self._utt_ids = np.array(utt_ids, dtype=f"S{_UTT_ID_WIDTH}")
        self._shard_idx = np.array(shard_idx, dtype=np.int32)
        self.durations = np.array(durations, dtype=np.float32)
        self._text_buffer = b"".join(text_chunks)
        self._text_offsets = np.array(offsets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._utt_ids.shape[0])

    def n_frames(self, hop_length: int, sample_rate: int) -> np.ndarray:
        """Analytic mel frame count, matching center-padded framing.

        ``1 + n_samples // hop`` is what torchaudio's centered STFT yields;
        ``duration * sr / hop`` is not the same thing.
        """
        n_samples = (self.durations.astype(np.float64) * sample_rate).astype(
            np.int64
        )
        return (1 + n_samples // hop_length).astype(np.int32)

    def _utt_id(self, idx: int) -> str:
        return self._utt_ids[idx].decode("ascii")

    def _text(self, idx: int) -> str:
        start = int(self._text_offsets[idx])
        end = int(self._text_offsets[idx + 1])
        return self._text_buffer[start:end].decode("utf-8")

    def _wav_path(self, idx: int) -> Path:
        shard = self._shards[int(self._shard_idx[idx])]
        return self.corpus_root / shard / (
            self._utt_id(idx) + self.audio_suffix
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        sample: dict[str, Any] = {"text": self._text(idx)}

        if self.load_speech:
            speech, speech_fs = sf.read(str(self._wav_path(idx)))
            speech = np.asarray(speech, dtype=np.float32)
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            if self.fs is not None and speech_fs != self.fs:
                speech = (
                    torchaudio.functional.resample(
                        torch.from_numpy(speech),
                        orig_freq=speech_fs,
                        new_freq=self.fs,
                    ).numpy().astype(np.float32)
                )
            sample["speech"] = speech

        if self.inference:
            sample.update({
                "utt_id": np.asarray(self._utt_id(idx)),
                "wav_path": str(self._wav_path(idx)),
                "raw_text": sample["text"],
            })
        return sample
