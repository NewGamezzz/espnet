"""Inference output helpers for the Emilia F5-TTS recipe.

Copied from egs3/libritts/tts/src/inference.py: SeedTTSDataset.__getitem__
(egs3/emilia/tts/dataset/seedtts.py) emits the same "utt_id" / "raw_text" /
"ref_wav_path" / "wav_path" keys the LibriTTS dataset does, so this maps in
unchanged. `ref` prefers `ref_wav_path` (the Seed-TTS prompt wav, F5's
cross-speaker protocol) and falls back to `wav_path` (the utterance's own
ground-truth wav) only for datasets that don't provide a separate prompt.
"""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    utt_id = data.get("utt_id", str(idx))
    text = str(data.get("raw_text", ""))
    # Speaker-similarity reference: the cross/same-speaker reference audio when
    # present (F5 cross-speaker protocol), else the utterance's own GT wav.
    ref = str(data.get("ref_wav_path") or data.get("wav_path", ""))
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("TTS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "text": text, "ref": ref, "wav": wav}
