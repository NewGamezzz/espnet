"""Assembly: embedding surgery, provenance check, builder, param groups."""

import hashlib
import json

import pytest
import torch
from .conftest import (
    BASE_TOKENS,
    EXT_TOKENS,
    FEATS_KWARGS,
    TINY_ARCH,
    make_dit,
)

from egs3.conversational.tts.dataset.preprocessing.text import (
    NEW_TOKENS,
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    make_token2id,
)
from egs3.conversational.tts.src.branch_exchange import (
    BranchContext,
    ExchangedBlock,
)
from egs3.conversational.tts.src.build_model import (
    build_multibranch_f5,
    check_vocab_provenance,
    exchange_param_groups,
    extended_text_embedding,
    load_pretrained_with_surgery,
)
from egs3.conversational.tts.src.multibranch_cfm import MultiBranchCFM
from espnet2.tts.f5.cfm import CFM

BUILD_CFM = dict(
    sigma=0.0,
    audio_drop_prob=0.0,
    cond_drop_prob=0.0,
    frac_lengths_mask=[0.7, 1.0],
)
BUILD_EXCHANGE = {"type": "tac", "schedule": {f"1-{TINY_ARCH['depth']}": "P+TAC"}}
_EMBED_KEY = "transformer.text_embed.text_embed.weight"


def build_tiny(ext_vocab_file, **overrides):
    kwargs = dict(
        vocab_file=str(ext_vocab_file),
        arch=TINY_ARCH,
        cfm=BUILD_CFM,
        exchange=BUILD_EXCHANGE,
        feats_extract=FEATS_KWARGS,
    )
    kwargs.update(overrides)
    return build_multibranch_f5(**kwargs)


def test_embedding_surgery_rows():
    dim = 16
    noise_scale = 0.01
    base_size = len(BASE_TOKENS)
    gen = torch.Generator().manual_seed(0)
    pretrained = torch.randn(base_size + 1, dim, generator=gen)

    new = extended_text_embedding(
        pretrained, EXT_TOKENS, noise_scale=noise_scale, generator=gen
    )

    assert new.shape == (len(EXT_TOKENS) + 1, dim)
    assert torch.isfinite(new).all()
    # Original rows (filler + every base token) are bit-equal.
    assert torch.equal(new[: base_size + 1], pretrained)

    token2id = make_token2id(EXT_TOKENS)
    warm_starts = {
        token2id[TURN_TOKEN] + 1: token2id[" "] + 1,  # <turn> <- space row
        token2id[OTHER_TOKEN] + 1: 0,  # <OTHER> <- filler row
        token2id[SPEAKER_PROMPT_TOKEN] + 1: 0,  # <speaker_prompt> <- filler row
        token2id[PREV_CHUNK_TOKEN] + 1: 0,  # <prev_chunk> <- filler row
        token2id[TURN_FILL_TOKEN] + 1: 0,  # <turn_fill> <- filler row
    }
    for row, source in warm_starts.items():
        diff = new[row] - pretrained[source]
        assert diff.abs().max() > 0  # noise was actually added
        # Within the noise scale (5 sigma per coordinate is generous).
        assert diff.abs().max() < 5 * noise_scale


def test_extended_embedding_four_new_rows():
    base = ["x", " ", "y"]
    tokens = base + list(NEW_TOKENS)
    weight = torch.randn(len(base) + 1, 8)
    out = extended_text_embedding(weight, tokens, noise_scale=0.0)
    assert out.shape == (len(tokens) + 1, 8)
    torch.testing.assert_close(out[: len(base) + 1], weight)
    space_row = base.index(" ") + 1
    torch.testing.assert_close(out[len(base) + 1], weight[space_row])  # <turn>
    torch.testing.assert_close(out[len(base) + 2], weight[0])  # <OTHER>
    torch.testing.assert_close(out[len(base) + 3], weight[0])  # <speaker_prompt>
    torch.testing.assert_close(out[len(base) + 4], weight[0])  # <prev_chunk>
    torch.testing.assert_close(out[len(base) + 5], weight[0])  # <turn_fill>


def test_turn_fill_row_warm_starts_from_filler():
    base = ["a", "b", " "]
    tokens = base + list(NEW_TOKENS)
    weight = torch.randn(len(base) + 1, 8)
    gen = torch.Generator().manual_seed(0)
    out = extended_text_embedding(weight, tokens, noise_scale=0.02, generator=gen)
    assert out.shape == (len(tokens) + 1, 8)
    tf_row = len(base) + 5  # token id len(base)+4, +1 for the filler shift
    assert torch.allclose(out[tf_row], weight[0], atol=0.1)
    assert not torch.equal(out[tf_row], weight[0])  # noise was added


def test_embedding_surgery_rejects_wrong_base():
    pretrained = torch.randn(len(BASE_TOKENS) + 3, 8)  # wrong row count
    with pytest.raises(ValueError, match="do not belong together"):
        extended_text_embedding(pretrained, EXT_TOKENS)


def test_pretrained_load_with_surgery(tmp_path):
    """Assembly step 2 on a fabricated 'pretrained' state dict: original
    embedding rows and ALL non-embedding weights are bit-equal."""
    base_cfm = CFM(
        transformer=make_dit(seed=3, text_num_embeds=len(BASE_TOKENS)),
        sigma=0.0,
        num_channels=FEATS_KWARGS["n_mels"],
    )
    ckpt = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": base_cfm.state_dict()}, ckpt)

    target = MultiBranchCFM(
        make_dit(seed=4),
        ctx=BranchContext(),
        sigma=0.0,
        num_channels=FEATS_KWARGS["n_mels"],
    )
    load_pretrained_with_surgery(
        target,
        ckpt,
        EXT_TOKENS,
        noise_scale=0.01,
        generator=torch.Generator().manual_seed(0),
    )

    pretrained_sd = base_cfm.state_dict()
    loaded_sd = target.state_dict()
    assert set(loaded_sd) == set(pretrained_sd)
    for key, value in pretrained_sd.items():
        if key == _EMBED_KEY:
            assert torch.equal(loaded_sd[key][: len(BASE_TOKENS) + 1], value)
            assert loaded_sd[key].shape[0] == len(EXT_TOKENS) + 1
        else:
            assert torch.equal(loaded_sd[key], value), key


def test_vocab_provenance(tmp_path):
    base_vocab = tmp_path / "base_vocab.txt"
    base_vocab.write_text("\n".join(BASE_TOKENS) + "\n", encoding="utf-8")
    meta = {
        "base_vocab_sha256": hashlib.sha256(base_vocab.read_bytes()).hexdigest(),
        "base_vocab_size": len(BASE_TOKENS),
    }
    meta_path = tmp_path / "vocab_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    check_vocab_provenance(meta_path, base_vocab)  # matching: no raise

    other_vocab = tmp_path / "other_vocab.txt"
    other_vocab.write_text("\n".join(BASE_TOKENS + ["z"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        check_vocab_provenance(meta_path, other_vocab)


def test_builder_rejects_non_extended_vocab(tmp_path):
    bad = tmp_path / "bad_vocab.txt"
    bad.write_text("\n".join(BASE_TOKENS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extended vocab"):
        build_multibranch_f5(
            vocab_file=str(bad),
            arch=TINY_ARCH,
            cfm=BUILD_CFM,
            exchange=BUILD_EXCHANGE,
            feats_extract=FEATS_KWARGS,
        )


def test_param_groups(ext_vocab_file):
    model = build_tiny(ext_vocab_file)
    groups = exchange_param_groups(model, lr_exchange=1e-4, lr_backbone=1e-5)

    injected = {
        id(p)
        for m in model.modules()
        if isinstance(m, ExchangedBlock)
        for p in m.exchange.parameters()
    }
    assert injected  # the schedule actually injected something
    exchange_ids = {id(p) for p in groups[0]["params"]}
    backbone_ids = {id(p) for p in groups[1]["params"]}

    # The exchange group contains exactly the injected modules' parameters
    # and nothing else; together the groups cover every trainable parameter.
    assert exchange_ids == injected
    assert exchange_ids.isdisjoint(backbone_ids)
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert exchange_ids | backbone_ids == all_ids
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-5


def test_builder_wires_mel_spec_kwargs(ext_vocab_file):
    """The CFM's internal MelSpec (raw-wave prompt path of CFM.sample) must
    match the training-time feats_extract, or the two mel front-ends
    silently diverge when the feats_extract block changes."""
    from espnet2.tts.f5.modules import get_vocos_mel_spectrogram

    model = build_tiny(ext_vocab_file)
    mel_spec = model.cfm.mel_spec
    extractor = model.feats_extract
    assert mel_spec.n_fft == extractor.n_fft
    assert mel_spec.hop_length == extractor.hop_length
    assert mel_spec.win_length == extractor.win_length
    assert mel_spec.n_mel_channels == extractor.n_mels
    assert mel_spec.target_sample_rate == extractor.fs
    # MelSpec keeps the type only as the bound extractor function.
    assert extractor.mel_spec_type == "vocos"
    assert mel_spec.extractor is get_vocos_mel_spectrogram


def test_builder_zero_init_gates(ext_vocab_file):
    model = build_tiny(ext_vocab_file)
    gates = [m.exchange.g for m in model.modules() if isinstance(m, ExchangedBlock)]
    assert len(gates) == TINY_ARCH["depth"]
    assert all(torch.equal(g, torch.zeros(())) for g in gates)
    # One shared context between the CFM and every injected block.
    assert all(
        m.ctx is model.cfm.ctx for m in model.modules() if isinstance(m, ExchangedBlock)
    )
