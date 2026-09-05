import random

import pytest
import torch
from src.model import DualPromptCFM, DualPromptF5TTS

from espnet3.systems.tts.f5_tts.f5tts import F5TTS

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
TINY = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    text_embedding_size=16,
    convolution_layers=1,
    feats_extract_config=dict(
        fs=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=100,
        mel_spec_type="vocos",
    ),
)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    speech = torch.randn(2, 24000 * 2, generator=g) * 0.1
    text = torch.tensor(
        [[5, 5, 5, 6, 6, 7, 2, 3] + [0] * 10, [5, 5, 7, 3, 2] + [0] * 13]
    )
    return dict(
        text=text,
        text_lengths=torch.tensor([8, 5]),
        speech=speech,
        speech_lengths=torch.tensor([48000, 40000]),
    )


def test_dual_prompt_model_is_f5tts_with_subclassed_cfm():
    m = DualPromptF5TTS(token_list=TOKENS, **TINY)
    assert isinstance(m, F5TTS) and isinstance(m.cfm, DualPromptCFM)
    assert m.cfm.transformer.dim == 32


def test_sentinel_is_bit_identical_to_stock():
    torch.manual_seed(0)
    stock = F5TTS(token_list=TOKENS, **TINY)
    torch.manual_seed(0)
    dual = DualPromptF5TTS(token_list=TOKENS, **TINY)
    dual.load_state_dict(stock.state_dict())
    b = _batch()
    torch.manual_seed(1)
    random.seed(1)
    l0, _, _ = stock(**b)
    torch.manual_seed(1)
    random.seed(1)
    l1, _, _ = dual(cond_frames=torch.tensor([[-1], [-1]]), **b)
    assert torch.equal(l0, l1)


def test_deterministic_span_masks_only_target():
    m = DualPromptF5TTS(token_list=TOKENS, **TINY)
    b = _batch()
    cf = torch.tensor([[100], [0]])
    feats, lens = m._extract_feats(b["speech"], b["speech_lengths"])
    mask = m.cfm.prediction_mask(lens, cf.view(-1))
    assert mask.shape == feats.shape[:2]
    assert not mask[0, :100].any() and mask[0, 100 : int(lens[0])].all()
    assert not mask[0, int(lens[0]) :].any()
    assert mask[1, : int(lens[1])].all() and not mask[1, int(lens[1]) :].any()
    loss, stats, _ = m(cond_frames=cf, **b)
    assert torch.isfinite(loss) and "loss" in stats


def test_cond_frames_beyond_length_raises():
    m = DualPromptF5TTS(token_list=TOKENS, **TINY)
    with pytest.raises(ValueError):
        m(cond_frames=torch.tensor([[10_000], [0]]), **_batch())
