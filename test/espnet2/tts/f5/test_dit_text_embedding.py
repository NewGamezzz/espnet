"""Tests for TextEmbedding's on-demand position-table growth (dit.py).

``freqs_cis`` starts at ``precompute_max_pos`` rows (upstream F5 hardcodes
8192, ~87.38 s of 24 kHz audio) and used to be a hard cap: any sequence
longer than the table crashed with a shape mismatch at the
``freqs * valid_pos_mask`` multiply. The table is a pure sinusoidal function
of the row index (non-persistent buffer, no learned state), so ``forward``
now grows it on demand; these tests pin the two properties that make that
safe: existing rows keep bit-identical values, and outputs for
already-supported lengths are unchanged by growth.
"""

import torch

from espnet2.tts.f5.backbones.dit import TextEmbedding
from espnet2.tts.f5.modules import precompute_freqs_cis


def _tiny_text_embed(initial_pos: int = 16) -> TextEmbedding:
    te = TextEmbedding(text_num_embeds=10, text_dim=8, conv_layers=1)
    # Shrink the initial table so growth triggers at test-sized lengths.
    te.precompute_max_pos = initial_pos
    te.freqs_cis = precompute_freqs_cis(8, initial_pos)
    return te.eval()


def test_sequence_longer_than_the_table_no_longer_crashes():
    te = _tiny_text_embed(initial_pos=16)
    text = torch.randint(0, 10, (2, 5))
    # Tensor seq_len exercises the valid_pos_mask multiply that crashed.
    out = te(text, seq_len=torch.tensor([40, 30]))
    assert out.shape == (2, 40, 8)


def test_growth_rounds_up_to_a_multiple_of_the_initial_size():
    te = _tiny_text_embed(initial_pos=16)
    te(torch.randint(0, 10, (1, 3)), seq_len=40)
    assert te.freqs_cis.shape[0] == 48  # ceil(40 / 16) * 16


def test_existing_positions_and_short_outputs_are_unchanged():
    torch.manual_seed(0)
    te = _tiny_text_embed(initial_pos=16)
    original_table = te.freqs_cis.clone()
    text = torch.randint(0, 10, (2, 5))
    with torch.no_grad():
        short_before = te(text, seq_len=10)
        te(text, seq_len=64)  # trigger growth
        short_after = te(text, seq_len=10)
    assert torch.equal(te.freqs_cis[:16], original_table)
    assert torch.equal(short_before, short_after)


def test_table_stays_a_registered_buffer_after_growth():
    te = _tiny_text_embed(initial_pos=16)
    te(torch.randint(0, 10, (1, 3)), seq_len=40)
    assert "freqs_cis" in dict(te.named_buffers())
    # Still non-persistent: growth must never leak into checkpoints.
    assert "freqs_cis" not in te.state_dict()
