import numpy as np
import pytest
import torch
from src.layout import (
    HOP,
    TokenTable,
    build_text_ids,
    cond_frames,
    n_frames_total,
    quantize_prompt_16k,
    region_frames,
)

TOKENS = [
    "<blank>",
    "<unk>",
    "a",
    "b",
    "<space>",
    "<spk>",
    "<lang>",
    "<de>",
    "<zh>",
    "<sos/eos>",
]


def _table(tmp_path):
    p = tmp_path / "tokens.txt"
    p.write_text("\n".join(TOKENS) + "\n")
    return TokenTable(p)


def test_frame_rules_match_vocoder_mel():
    from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec

    mel = VocoderMelSpec(
        fs=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=100,
        mel_spec_type="vocos",
    )
    for n in (24000, 24000 + 255, 3 * 256 * 100):
        _, lens = mel(torch.zeros(1, n), torch.tensor([n]))
        assert int(lens[0]) == n_frames_total(n)
    assert region_frames(768) == 3
    with pytest.raises(AssertionError):
        region_frames(700)


def test_quantized_prompt_resamples_to_hop_multiple():
    n16 = quantize_prompt_16k(16000 + 300)
    assert n16 % 512 == 0 and (n16 * 3 // 2) % HOP == 0


def test_text_ids_golden(tmp_path):
    t = _table(tmp_path)
    ids = build_text_ids(3, 2, "de", ["a", "b", "<space>", "zz"], t)
    assert ids.tolist() == [5, 5, 5, 6, 6, 7, 2, 3, 4, 1]  # unknown 'zz' -> <unk>
    assert ids.dtype == np.int64
    assert cond_frames(3, 2) == 5
    assert build_text_ids(0, 0, "zh", ["a"], t).tolist() == [8, 2]
    assert t.size == len(TOKENS) and t.spk == 5 and t.lang == 6
