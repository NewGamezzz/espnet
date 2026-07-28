"""Tests for the self-contained rotary embedding port (espnet2/tts/f5/rotary.py).

x_transformers decorates ``RotaryEmbedding.forward`` and ``apply_rotary_pos_emb``
with ``@autocast(enabled=False)``: rope math must run in fp32 even under AMP.
bf16 cannot represent positions above 256 exactly (spacing 16 in [2048, 4096]),
so if autocast lowers the position/inv_freq einsum to bf16, positional phases
silently corrupt, growing with utterance length. CUDA and MPS autocast both
lower einsum to the autocast dtype (CPU autocast does not, so these tests need
a real accelerator).
"""

import pytest
import torch

from espnet2.tts.f5.rotary import RotaryEmbedding, apply_rotary_pos_emb


def _amp_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


AMP_DEVICE = _amp_device()

needs_amp_device = pytest.mark.skipif(
    AMP_DEVICE is None,
    reason="needs cuda or mps: cpu autocast does not lower einsum",
)


@needs_amp_device
def test_rope_freqs_under_autocast_match_fp32():
    rope = RotaryEmbedding(dim=64).to(AMP_DEVICE)
    seq_len = 4096

    ref_freqs, _ = rope.forward_from_seq_len(seq_len)

    with torch.autocast(device_type=AMP_DEVICE, dtype=torch.bfloat16):
        amp_freqs, _ = rope.forward_from_seq_len(seq_len)

    assert amp_freqs.dtype == torch.float32
    torch.testing.assert_close(amp_freqs, ref_freqs, rtol=0.0, atol=0.0)


@needs_amp_device
def test_apply_rotary_under_autocast_matches_fp32_reference():
    rope = RotaryEmbedding(dim=64).to(AMP_DEVICE)
    seq_len = 4096
    torch.manual_seed(0)
    q = torch.randn(1, seq_len, 64, device=AMP_DEVICE)

    ref_freqs, _ = rope.forward_from_seq_len(seq_len)
    ref = apply_rotary_pos_emb(q, ref_freqs)

    with torch.autocast(device_type=AMP_DEVICE, dtype=torch.bfloat16):
        amp_freqs, _ = rope.forward_from_seq_len(seq_len)
        out = apply_rotary_pos_emb(q.to(torch.bfloat16), amp_freqs)

    # Only q's own bf16 rounding (~0.4% relative) is tolerated; corrupted
    # phases at positions > 256 produce O(1) errors and must fail.
    torch.testing.assert_close(out.float(), ref, rtol=0.0, atol=0.05)
