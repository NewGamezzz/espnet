"""LibriTTS dataset backed by recipe TSV manifests."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torchaudio
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset as TorchDataset

from egs3.libritts.tts.dataset.builder import LibriTTSBuilder
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    wav_path: Path
    text: str
    sid: int


def _read_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text, sid = line.split("\t", maxsplit=3)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                    sid=int(sid),
                )
            )
    if not entries:
        raise RuntimeError(f"Manifest is empty: {path}")
    return entries


class LibriTTSDataset(TorchDataset):
    """LibriTTS dataset returning text/speech/spembs samples.

    The output keys match what VITS via ``GANTTSTask`` consumes:
      - ``text``  : raw transcript string (tokenized later by ``CommonPreprocessor``)
      - ``speech``: float32 waveform
      - ``spembs``: float32 speaker embedding (x-vector) loaded from a ``.pt`` file
    
    The dataset cosumnes a following argument during initialization:
        - ``split``: A string key for the dataset split
        - ``recipe_dir``: Optional path to the recipe root, used to resolve the default
        - ``manifest_path``: Optional path to the manifest TSV file. If not supplied, the dataset
            will look for the default manifest path for the given split in the recipe's data directory.
        - ``load_speech``: Whether to load the speech waveform from disk. If False
            the sample will not include the "speech" key. Default: True.
        - ``load_xvector``: Whether to load the speaker embedding (x-vector) from
            disk. If False the sample will not include the "spembs" key. Default: True.
        - ``xvector_dir``: Optional path to the directory containing x-vector .pt files
            (one per utterance, named {utt_id}.pt). Required if ``load_xvector`` is True.
        - ``fs``: Optional target sampling rate for the speech waveform. If supplied,
            the waveform will be resampled to this rate after loading. Default: None (no resampling).
        - ``inference``: If True, the dataset will include additional metadata in each
            sample for inference purposes (utt_id, wav_path, raw_text). Default: False.
        - ``ref_mode``: If not None, include a reference utterance in each sample
            for zero-shot voice cloning inference. Must be one of "same_speaker" or
            "cross_speaker". Default: None (no reference).
        - ``ref_seed``: Random seed for deterministic reference selection when ``ref_mode`` is not None. Default: 0.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        load_xvector: bool = True,
        xvector_dir: str | Path | None = None,
        fs: int | None = None,
        inference: bool = False,
        ref_mode: str | None = None,
        ref_seed: int = 0,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        self.load_xvector = load_xvector
        self.inference = inference
        self.fs = fs
        if ref_mode not in (None, "same_speaker", "cross_speaker"):
            raise ValueError(
                "ref_mode must be None, 'same_speaker', or 'cross_speaker', "
                f"got {ref_mode!r}."
            )
        self.ref_mode = ref_mode
        self.ref_seed = ref_seed
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        if self.load_xvector:
            if xvector_dir is None:
                raise ValueError(
                    "xvector_dir must be supplied when load_xvector is True. "
                    "Pass it via data_src_args in training.yaml, e.g. "
                    "xvector_dir: ${xvector.save_path}/${xvector.spk_embed_tag}_train"
                )
            self.xvector_dir = Path(xvector_dir)
            if not self.xvector_dir.is_dir():
                raise FileNotFoundError(
                    f"xvector_dir does not exist: {self.xvector_dir}. "
                    "Run compute_xvectors stage first."
                )
        else:
            self.xvector_dir = None

        builder = LibriTTSBuilder()
        if not builder.is_built(recipe_dir=recipe_root):
            raise RuntimeError(
                "Dataset is not built yet. Run create_dataset stage first."
            )

        # Caller-supplied manifest_path wins. Otherwise fall back to the
        # split-keyed default from dataset/config.yaml (unfiltered manifest).
        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = (recipe_root / resolved_manifest).resolve()
        else:
            if split not in _SPLIT_MANIFEST_PATHS:
                raise ValueError(
                    f"Unknown split '{split}'. Expected one of "
                    f"{sorted(_SPLIT_MANIFEST_PATHS)}"
                )
            resolved_manifest = self.data_dir / _SPLIT_MANIFEST_PATHS[split]
        if not resolved_manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {resolved_manifest}")
        self.manifest_path = resolved_manifest
        self._entries = _read_manifest(resolved_manifest)

        # Precompute a deterministic reference utterance per entry for zero-shot
        # voice-cloning inference (F5-TTS). The reference is drawn from the same
        # split: a different utterance of the same speaker ("same_speaker") or an
        # utterance of a different speaker ("cross_speaker"). F5 needs the
        # reference transcript, so __getitem__ returns ref_speech + ref_text.
        self._ref_idx: list[int] | None = None
        if self.ref_mode is not None:
            self._ref_idx = self._build_ref_index()

    def __len__(self) -> int:
        return len(self._entries)

    def _build_ref_index(self) -> list[int]:
        """Pick one reference entry index per target, deterministically."""
        rng = random.Random(self.ref_seed)
        by_spk: dict[int, list[int]] = defaultdict(list)
        for idx, entry in enumerate(self._entries):
            by_spk[entry.sid].append(idx)
        speakers = list(by_spk.keys())

        ref_idx: list[int] = []
        for i, entry in enumerate(self._entries):
            if self.ref_mode == "same_speaker":
                cands = [j for j in by_spk[entry.sid] if j != i]
                choice = rng.choice(cands) if cands else i
            else:  # cross_speaker
                other_spks = [s for s in speakers if s != entry.sid]
                if other_spks:
                    spk = rng.choice(other_spks)
                    choice = rng.choice(by_spk[spk])
                else:  # single-speaker split: fall back to self
                    choice = i
            ref_idx.append(choice)
        return ref_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        # utt_id, wav_path, and raw_text are included for inference purposes only.
        sample: dict[str, Any] = {
            "text": entry.text,
        }
        if self.load_speech:
            speech, speech_fs = sf.read(str(entry.wav_path))
            sample["speech"] = np.asarray(speech, dtype=np.float32)
            if self.fs is not None and speech_fs != self.fs:
                # Resample if a target sampling rate is specified and different from the original.
                sample["speech"] = torchaudio.functional.resample(
                    torch.from_numpy(sample["speech"]), orig_freq=speech_fs, new_freq=self.fs
                ).numpy().astype(np.float32)
        if self.load_xvector:
            pt_path = self.xvector_dir / f"{entry.utt_id}.pt"
            if not pt_path.is_file():
                raise FileNotFoundError(f"X-vector missing: {pt_path}")
            spembs = torch.load(str(pt_path), map_location="cpu")
            if isinstance(spembs, torch.Tensor):
                spembs = spembs.numpy()
            sample["spembs"] = np.asarray(spembs, dtype=np.float32).squeeze()
        if self._ref_idx is not None:
            ref_entry = self._entries[self._ref_idx[int(idx)]]
            ref_speech, _ = sf.read(str(ref_entry.wav_path))
            sample["ref_speech"] = np.asarray(ref_speech, dtype=np.float32)
            sample["ref_text"] = ref_entry.text
            if self.inference:
                # ref_wav_path is the speaker-similarity reference for metrics.
                sample["ref_wav_path"] = str(ref_entry.wav_path)
                sample["ref_utt_id"] = np.asarray(ref_entry.utt_id)
        if self.inference:
            # For inference, we additionally return utt_id,
            # raw_text, and wav_path for metrics calculation.
            metadata = {
                "utt_id": np.asarray(entry.utt_id),
                "wav_path": str(entry.wav_path),
                "raw_text": entry.text,
            }
            sample.update(metadata)
        return sample
