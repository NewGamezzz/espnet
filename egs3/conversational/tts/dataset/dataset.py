"""Conversation dataset backed by window manifests, plus the packed collator.

Audio is seek-read from the session FLAC (only the window's segment), kept
per-channel, and resampled 48 -> 24 kHz on the fly.  Batches use the packed
layout of the ``branch_exchange`` package: a per-conversation ``counts`` list
plus row-stacked ``(sum(counts), ...)`` tensors with no padding rows on the
branch axis, so mixed channel counts per batch need no special casing.
"""

from __future__ import annotations

import json
import os
import random
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset

from espnet2.fileio.sound_scp import soundfile_read
from espnet3.utils.config_utils import load_config_with_defaults

from .text import build_branch_texts, encode_tokens, make_token2id
from .windows import WindowRecord, from_json

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]


def resolve_dataset_root(explicit: str | Path | None = None) -> Path:
    """Corpus root resolution used across the recipe: explicit argument >
    ``$SSSD_ROOT`` > ``builder.dataset_root`` in config.yaml."""
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("SSSD_ROOT"):
        return Path(os.environ["SSSD_ROOT"])
    return Path(_BUILDER_CFG["dataset_root"])


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


def read_vocab(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.rstrip("\n")]


class ConversationDataset(TorchDataset):
    """Multi-channel conversation windows for F5-TTS fine-tuning.

    Each item:
      - ``window_id``    : str
      - ``num_channels`` : int, N of this conversation
      - ``speech``       : float32 tensor (N, T) at ``fs`` (default 24 kHz)
      - ``text``         : list of N variable-length int64 tensors (no padding;
                           the model pads text up to the mel length itself)
      - ``perm``         : int64 tensor (N,), the channel permutation applied
                           to both audio rows and text sequences (row k holds
                           original channel ``perm[k]``)

    ``permute_channels`` defaults to ``split == "train"``; it guards against
    systematic ch0/ch1 artifacts in the corpus and is applied consistently to
    audio and texts (turn markers carry no identity, so nothing else needs
    re-indexing).
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
        vocab_path: str | Path | None = None,
        fs: int | None = None,
        permute_channels: bool | None = None,
        seed: int = 0,
        inference: bool = False,
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
        if vocab_path is None:
            vocab_path = data_dir / _DATASET_CFG["vocab_path"]
        manifest_path = Path(manifest_path)
        vocab_path = Path(vocab_path)
        for path, hint in ((manifest_path, "window manifest"), (vocab_path, "vocab")):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{hint} not found: {path}. Run the SSSD builder first "
                    "(python -m egs3.conversational.tts.dataset.builder)."
                )
        self.records = read_window_manifest(manifest_path)
        self.token2id = make_token2id(read_vocab(vocab_path))

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
                raise ValueError(f"fixed perm {perm} is not a permutation of range({n})")
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
        branch_tokens = build_branch_texts(record.turns, n)
        perm = self._draw_perm(n)
        sample: dict[str, Any] = {
            "window_id": record.window_id,
            "num_channels": n,
            "speech": speech[perm],
            "text": [
                torch.tensor(
                    encode_tokens(branch_tokens[p], self.token2id), dtype=torch.long
                )
                for p in perm
            ],
            "perm": torch.tensor(perm, dtype=torch.long),
        }
        if self.inference:
            inv = {orig: row for row, orig in enumerate(perm)}
            sample.update(
                session_id=record.session_id,
                t0=record.t0,
                t1=record.t1,
                audio_path=str(self.dataset_root / record.audio_relpath),
                turns=[
                    {
                        "channel": inv[t.channel],  # row index after permutation
                        "speaker": t.speaker,
                        "text": t.text,
                        "start": t.start,
                        "end": t.end,
                    }
                    for t in record.turns
                ],
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
