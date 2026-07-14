"""
F5-TTS Dataset and DataLoader for ESPnet.

Matches the output dict of the reference collate_fn:
    {
        "mel":          (B, n_mel_channels, T_max)   float32
        "mel_lengths":  (B,)                          int64
        "text":         List[str]                     raw text, length B
        "text_lengths": (B,)                          int64  (chars per utterance)
    }

Reference:
  https://github.com/SWivid/F5-TTS/blob/2ae2c9bd9b64dab2cb069c4b97e5e7673c521e01/src/f5_tts/model/dataset.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


class MelSpec(nn.Module):
    """
    Mel spectrogram extractor matching F5-TTS defaults.

    Default parameters (vocos variant):
        sample_rate  = 24 000 Hz
        n_fft        = 1 024
        hop_length   = 256
        win_length   = 1 024
        n_mels       = 100
    """

    def __init__(
        self,
        target_sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        win_length: int = 1_024,
        n_mel_channels: int = 100,
    ):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mel_channels,
            power=1.0,          # amplitude, same as vocos convention
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (1, T) waveform
        Returns:
            mel: (n_mel_channels, T') spectrogram
        """
        mel = self.mel_transform(audio)         # (1, n_mels, T')
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)                   # (n_mels, T')


class F5TTSDataset(Dataset):
    """
    Reads a pipe-delimited CSV (audio_file|text) produced by prepare_emilia.py,
    loads waveforms on the fly, and computes mel spectrograms.

    Skips utterances outside [min_duration, max_duration] seconds (same
    guard as the reference CustomDataset).
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        win_length: int = 1_024,
        n_mel_channels: int = 100,
        min_duration: float = 0.3,
        max_duration: float = 30.0,
        durations: Optional[List[float]] = None,
        mel_spec_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.mel_spectrogram = mel_spec_module or MelSpec(
            target_sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
        )

        # Load CSV
        csv_path = Path(csv_path)
        self.samples: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                self.samples.append(
                    {"audio_file": row["audio_file"].strip(), "text": row["text"].strip()}
                )

        # Optional pre-loaded durations (from duration.json) for DynamicBatchSampler
        self.durations = durations

        # Resampler cache — created lazily per source sample rate
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def get_frame_len(self, index: int) -> float:
        """Return approximate number of mel frames for utterance `index`."""
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        # Fall back: read duration from file metadata (slow — prefer duration.json)
        info = torchaudio.info(self.samples[index]["audio_file"])
        duration = info.num_frames / info.sample_rate
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        # Skip utterances outside duration bounds, same as reference
        while True:
            sample = self.samples[index]
            audio_path = sample["audio_file"]
            text = sample["text"]

            audio, source_sr = torchaudio.load(audio_path)
            duration = audio.shape[-1] / source_sr

            if self.min_duration <= duration <= self.max_duration:
                break
            index = (index + 1) % len(self)

        # Mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if needed
        if source_sr != self.target_sample_rate:
            if source_sr not in self._resamplers:
                self._resamplers[source_sr] = torchaudio.transforms.Resample(
                    source_sr, self.target_sample_rate
                )
            audio = self._resamplers[source_sr](audio)

        # Mel spectrogram — (n_mels, T')
        mel = self.mel_spectrogram(audio)   # MelSpec.forward handles squeeze

        return {
            "mel_spec": mel,    # (n_mels, T')
            "text": text,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pads mel spectrograms to the longest in the batch.

    Output dict (identical keys to F5-TTS reference):
        mel:          (B, n_mels, T_max)   float32
        mel_lengths:  (B,)                  int64
        text:         List[str]             length B
        text_lengths: (B,)                  int64
    """
    mel_specs = [item["mel_spec"] for item in batch]            # each (n_mels, T_i)
    mel_lengths = torch.LongTensor([s.shape[-1] for s in mel_specs])
    max_mel_len = mel_lengths.amax()

    padded_mels = []
    for spec in mel_specs:
        pad_len = max_mel_len - spec.shape[-1]
        padded_mels.append(F.pad(spec, (0, pad_len), value=0.0))

    mel = torch.stack(padded_mels)                              # (B, n_mels, T_max)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(t) for t in text])

    return dict(
        mel=mel,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )

class DynamicBatchSampler(Sampler):
    """
    Bins utterances by length and fills each batch up to `frames_threshold`
    total mel frames, minimising padding waste.

    Mirrors DynamicBatchSampler from F5-TTS dataset.py.
    """

    def __init__(
        self,
        dataset: F5TTSDataset,
        frames_threshold: int,
        max_samples: int = 0,
        random_seed: Optional[int] = None,
        drop_residual: bool = False,
    ):
        self.dataset = dataset
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        # Sort indices by frame length
        indices = [
            (i, dataset.get_frame_len(i))
            for i in tqdm(range(len(dataset)), desc="Building dynamic batches")
        ]
        indices.sort(key=lambda x: x[1])

        batches: List[List[int]] = []
        batch: List[int] = []
        batch_frames = 0.0

        for idx, frame_len in indices:
            fits = batch_frames + frame_len <= frames_threshold
            under_max = max_samples == 0 or len(batch) < max_samples

            if fits and under_max:
                batch.append(idx)
                batch_frames += frame_len
            else:
                if batch:
                    batches.append(batch)
                if frame_len <= frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0.0

        if not drop_residual and batch:
            batches.append(batch)

        self.batches = batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            order = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in order]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def build_dataloader(
    csv_path: str | Path,
    frames_threshold: int = 45_000,
    duration_json_path: Optional[str | Path] = None,
    target_sample_rate: int = 24_000,
    n_fft: int = 1_024,
    hop_length: int = 256,
    win_length: int = 1_024,
    n_mel_channels: int = 100,
    num_workers: int = 4,
    random_seed: int = 42,
    max_samples: int = 0,
    drop_residual: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader with DynamicBatchSampler from a CSV manifest.

    Args:
        csv_path:             path to train.csv or val.csv
        frames_threshold:     max total mel frames per batch (tune to GPU VRAM)
        duration_json_path:   path to duration.json; avoids loading audio just
                              to get lengths when building the sampler
        num_workers:          DataLoader worker processes
        random_seed:          seed for per-epoch shuffle in DynamicBatchSampler
    """
    durations = None
    if duration_json_path is not None:
        with open(duration_json_path, "r", encoding="utf-8") as f:
            durations = json.load(f)["duration"]

    dataset = F5TTSDataset(
        csv_path=csv_path,
        target_sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        durations=durations,
    )

    sampler = DynamicBatchSampler(
        dataset=dataset,
        frames_threshold=frames_threshold,
        max_samples=max_samples,
        random_seed=random_seed,
        drop_residual=drop_residual,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )