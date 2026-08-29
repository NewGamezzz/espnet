"""The DiT text branch's sinusoidal position table is sized by a constructor
argument, so long Chorus windows (120 s = 11,250 mel frames) fit.

Regression for smoke 3042228 (2026-08-28), which died with a raw broadcast
mismatch ("tensor a (8192) must match tensor b (9278)") on the first step.
"""

import pytest
import torch

from espnet2.tts.f5.backbones.dit import TextEmbedding

TEXT_DIM = 32
LONG_FRAMES = 9278  # the length that crashed the smoke


def _embed(max_pos):
    torch.manual_seed(0)
    return TextEmbedding(
        text_num_embeds=64, text_dim=TEXT_DIM, conv_layers=1, precompute_max_pos=max_pos
    )


def test_default_is_unchanged_at_8192():
    assert _embed(8192).precompute_max_pos == TextEmbedding(
        text_num_embeds=64, text_dim=TEXT_DIM, conv_layers=1
    ).precompute_max_pos == 8192


def test_sample_longer_than_the_table_raises_a_clear_error():
    embed = _embed(8192)
    text = torch.randint(0, 64, (1, 16))
    with pytest.raises(ValueError, match="exceeds the text position table"):
        embed(text, seq_len=torch.tensor([LONG_FRAMES]))


def test_raised_table_accepts_the_long_sample():
    embed = _embed(16384)
    text = torch.randint(0, 64, (1, 16))
    out = embed(text, seq_len=torch.tensor([LONG_FRAMES]))
    assert out.shape == (1, LONG_FRAMES, TEXT_DIM)
    assert torch.isfinite(out).all()


def test_raising_the_table_does_not_change_short_sample_output():
    small, large = _embed(8192), _embed(16384)
    large.load_state_dict(small.state_dict())
    text = torch.randint(0, 64, (2, 12))
    seq_len = torch.tensor([300, 180])
    small.eval(), large.eval()
    with torch.no_grad():
        assert torch.equal(small(text, seq_len=seq_len), large(text, seq_len=seq_len))


def test_table_is_not_a_checkpoint_key():
    assert "freqs_cis" not in _embed(16384).state_dict()
