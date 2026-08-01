"""Conversation dataset backed by window manifests, plus the packed collator.

Audio is seek-read from the session FLAC (only the window's segment), kept
per-channel, and resampled 48 -> 24 kHz on the fly.  The dataset is
vocab-agnostic: items carry the raw turns and tokenization happens in
``preprocessor.ConversationalTextPreprocessor`` (the ``DataOrganizer``
``preprocessor:`` slot), which fills in the ``text`` key the collator packs.
Batches use the packed layout of the ``branch_exchange`` package: a
per-conversation ``counts`` list plus row-stacked ``(sum(counts), ...)``
tensors with no padding rows on the branch axis, so mixed channel counts per
batch need no special casing.
"""

from __future__ import annotations

import dataclasses
import json
import random
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset

from espnet2.fileio.sound_scp import soundfile_read
from espnet3.utils.config_utils import load_config_with_defaults

from .builder import resolve_dataset_root
from .preprocessing.windows import WindowRecord, from_json

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]


def read_window_manifest(path: str | Path) -> list[WindowRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(from_json(json.loads(line)))
    if not records:
        raise RuntimeError(f"Window manifest is empty: {path}")
    return records


class ConversationDataset(TorchDataset):
    """Multi-channel conversation windows for F5-TTS fine-tuning.

    Each item:
      - ``window_id``    : str
      - ``num_channels`` : int, N of this conversation
      - ``speech``       : float32 tensor (N, T) at ``fs`` (default 24 kHz)
      - ``turns``        : the window's ``Turn`` records in conversation
                           order, with ``channel`` remapped to the
                           post-permutation row index (``speaker``/``text``/
                           ``start``/``end`` kept verbatim)
      - ``perm``         : int64 tensor (N,), the channel permutation applied
                           to the audio rows (row k holds original channel
                           ``perm[k]``)

    No token ids here: tokenization lives in the recipe preprocessor
    (``preprocessor.ConversationalTextPreprocessor``), which derives the
    per-branch ``text`` tensors from ``turns``.  Because ``channel`` is
    pre-remapped, everything downstream is permutation-agnostic.

    ``permute_channels`` defaults to ``split == "train"``; it guards against
    systematic ch0/ch1 artifacts in the corpus and is applied consistently to
    audio rows and turn channels (turn markers carry no identity, so nothing
    else needs re-indexing).

    ``min_active_speakers`` drops windows with fewer active speakers (the
    manifests keep single-speaker windows; the POC trains with ``2`` so
    gradient mass concentrates on actual interactions - a knob, not a
    rebuild).
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
        fs: int | None = None,
        permute_channels: bool | None = None,
        seed: int = 0,
        inference: bool = False,
        min_active_speakers: int = 1,
    ) -> None:
        self.split = split
        self.fs = int(fs if fs is not None else _DATASET_CFG["sample_rate"])
        self.permute_channels = (
            permute_channels if permute_channels is not None else split == "train"
        )
        self.seed = seed
        self.inference = inference
        self.dataset_root = resolve_dataset_root(dataset_root)

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        data_dir = recipe_root / _BUILDER_CFG["data_path"]
        if manifest_path is None:
            if split not in _DATASET_CFG["split_manifest_paths"]:
                raise KeyError(f"unknown split {split!r} and no manifest_path given")
            manifest_path = data_dir / _DATASET_CFG["split_manifest_paths"][split]
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"window manifest not found: {manifest_path}. Run the "
                "corresponding dataset builder first (SSSD: python -m "
                "egs3.conversational.tts.dataset.builder; LibriTTS: python -m "
                "egs3.conversational.tts.dataset.libritts_builder)."
            )
        self.records = read_window_manifest(manifest_path)
        if min_active_speakers > 1:
            self.records = [
                r for r in self.records if r.num_active_speakers >= min_active_speakers
            ]
            if not self.records:
                raise RuntimeError(
                    f"no window in {manifest_path} has >= "
                    f"{min_active_speakers} active speakers"
                )

        # Test hook: a fixed permutation (sequence of ints) overrides the RNG.
        self._fixed_perm: Sequence[int] | None = None
        # Created lazily on first __getitem__ so each DataLoader worker gets
        # its own stream (torch re-seeds workers per epoch); creating it here
        # would fork one shared state into every worker.
        self._worker_rng: random.Random | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _draw_perm(self, n: int) -> list[int]:
        if self._fixed_perm is not None:
            perm = list(self._fixed_perm)
            if sorted(perm) != list(range(n)):
                raise ValueError(
                    f"fixed perm {perm} is not a permutation of range({n})"
                )
            return perm
        if not self.permute_channels:
            return list(range(n))
        if self._worker_rng is None:
            self._worker_rng = random.Random(f"{self.seed}:{torch.initial_seed()}")
        return self._worker_rng.sample(range(n), n)

    def _load_speech(self, record: WindowRecord) -> torch.Tensor:
        path = self.dataset_root / record.audio_relpath
        start = round(record.t0 * record.sample_rate)
        stop = round(record.t1 * record.sample_rate)
        array, rate = soundfile_read(
            str(path), dtype="float32", start=start, end=stop, always_2d=True
        )
        if rate != record.sample_rate:
            raise RuntimeError(
                f"{path}: sample rate {rate} != manifest rate {record.sample_rate}"
            )
        if array.shape[1] != record.num_channels:
            raise RuntimeError(
                f"{path}: {array.shape[1]} channels != manifest {record.num_channels}"
            )
        speech = torch.from_numpy(array.T.copy())  # (N, T_src)
        if self.fs != record.sample_rate:
            speech = torchaudio.functional.resample(
                speech, orig_freq=record.sample_rate, new_freq=self.fs
            )
        return speech

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        n = record.num_channels
        speech = self._load_speech(record)
        perm = self._draw_perm(n)
        inv = {orig: row for row, orig in enumerate(perm)}
        sample: dict[str, Any] = {
            "window_id": record.window_id,
            "num_channels": n,
            "speech": speech[perm],
            "turns": [
                dataclasses.replace(t, channel=inv[t.channel])  # row index
                for t in record.turns
            ],
            "perm": torch.tensor(perm, dtype=torch.long),
        }
        if self.inference:
            sample.update(
                session_id=record.session_id,
                t0=record.t0,
                t1=record.t1,
                audio_path=str(self.dataset_root / record.audio_relpath),
            )
        return sample


def collate_conversations(
    samples: list[dict[str, Any]], *, text_pad_value: int | None = None
) -> dict[str, Any]:
    """Pack a batch of conversations into the branch_exchange layout.

    Rows are conversation-contiguous (sample order, branch order within each
    sample), matching ``conv_id = arange(B).repeat_interleave(counts)`` as
    built by ``BranchContext.branches(counts)``.  Audio pads with 0.0 plus a
    validity mask; text pads with -1 (the F5 model shifts ids by +1 and treats
    0 as its internal filler).
    """
    if text_pad_value is None:
        text_pad_value = int(_DATASET_CFG["text_pad_value"])
    counts = [s["num_channels"] for s in samples]
    m = sum(counts)
    speech_rows = [row for s in samples for row in s["speech"]]
    text_rows = [t for s in samples for t in s["text"]]

    t_max = max(row.shape[0] for row in speech_rows)
    speech = torch.zeros(m, t_max, dtype=torch.float32)
    speech_mask = torch.zeros(m, t_max, dtype=torch.bool)
    speech_lengths = torch.zeros(m, dtype=torch.long)
    for i, row in enumerate(speech_rows):
        speech[i, : row.shape[0]] = row
        speech_mask[i, : row.shape[0]] = True
        speech_lengths[i] = row.shape[0]

    l_max = max(t.shape[0] for t in text_rows)
    text = torch.full((m, l_max), text_pad_value, dtype=torch.long)
    text_lengths = torch.zeros(m, dtype=torch.long)
    for i, t in enumerate(text_rows):
        text[i, : t.shape[0]] = t
        text_lengths[i] = t.shape[0]

    return {
        "counts": counts,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "speech_mask": speech_mask,
        "text": text,
        "text_lengths": text_lengths,
        "window_ids": [s["window_id"] for s in samples],
    }
