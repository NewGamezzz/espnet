"""UTMOS wrapper (Task 5): predicted mean opinion score for synthesized
speech naturalness.

`utmos()` wraps SpeechMOS's `utmos22_strong` `torch.hub` model - the
heavy, GPU-bound import (`torch`, and the `torch.hub.load` network fetch
on first use) happens lazily, inside the function only, so importing this
module never pulls in `torch`. The loaded model is cached in a module
global because loading it is expensive and Task 8 calls `utmos()` once
per generated eval window.
"""

from __future__ import annotations

import soundfile as sf

_model = None


def utmos(wav_path: str, device: str = "cpu") -> float:
    """Predict UTMOS naturalness MOS for `wav_path`.

    Lazily imports `torch` and loads `tarepan/SpeechMOS:v1.2.0`'s
    `utmos22_strong` via `torch.hub.load(..., trust_repo=True)` (never at
    module scope), caching it in a module global so repeat calls pay the
    load cost once. Reads `wav_path` with `soundfile`, folds to mono by
    averaging channels, and scores `(tensor[1, T], sr)` on `device`.
    """
    import torch

    global _model
    if _model is None:
        _model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )

    model = _model.to(device)

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype("float32")
    tensor = torch.from_numpy(mono).unsqueeze(0).to(device)

    score = model(tensor, sr)
    return float(score.item())
