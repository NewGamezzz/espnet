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


def test_extended_embedding_five_new_rows():
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


def test_builder_passes_severed_to_every_tac(ext_vocab_file):
    """Table 4 row c: `exchange.severed: true` reaches every injected TAC."""
    model = build_tiny(ext_vocab_file, exchange={**BUILD_EXCHANGE, "severed": True})
    flags = [m.exchange.severed for m in model.modules() if isinstance(m, ExchangedBlock)]
    assert flags == [True] * TINY_ARCH["depth"]
    model = build_tiny(ext_vocab_file)
    flags = [m.exchange.severed for m in model.modules() if isinstance(m, ExchangedBlock)]
    assert flags == [False] * TINY_ARCH["depth"]


def test_severed_is_a_config_property_not_a_checkpoint_key(ext_vocab_file):
    """Characterization guard for a silent-corruption trap in the Table 4 rows.

    `severed` changes no parameter, so the two rows' state dicts are
    identical in keys and shapes and `load_state_dict(strict=True)` cannot
    tell them apart. Row c's weights loaded through row d's config would
    therefore run a live average path they were never trained with, with no
    error. The eval config's `train_config` MUST name the severed training
    yaml; this test fails if a future change makes `severed` a buffer or
    parameter, at which point the guard can be relaxed.
    """
    tac = build_tiny(ext_vocab_file)
    severed = build_tiny(ext_vocab_file, exchange={**BUILD_EXCHANGE, "severed": True})
    a, b = tac.state_dict(), severed.state_dict()
    assert a.keys() == b.keys()
    assert all(a[k].shape == b[k].shape for k in a)
    severed.load_state_dict(a, strict=True)  # no error: the trap
    assert all(
        m.exchange.severed for m in severed.modules() if isinstance(m, ExchangedBlock)
    )


def test_coattention_is_parameter_matched_to_tac(ext_vocab_file):
    """Row e must stay capacity-matched to rows c and d, since Table 4 prints
    a parameter column and the whole ablation rests on matched capacity.

    With `d_c` omitted the attention projections default to the model width,
    so both blocks cost 4 d^2. The residual is exactly one LayerNorm (2 d)
    plus one extra bias vector (d), less TAC's three PReLU scalars, i.e.
    3 d - 3 per block for any width. Asserting that closed form rather than a
    tolerance keeps the test meaningful at every dimension: the overhead is
    linear in d while the block is quadratic, so the RELATIVE gap shrinks as
    the model grows (0.18% at d=32, 0.073% at the real d=1024).

    This fails if someone adds a bottleneck or otherwise resizes the block.
    """
    from egs3.conversational.tts.src.branch_exchange import (
        BranchMHAExchange,
        TACExchange,
    )

    for dim, heads in ((32, 8), (256, 8), (1024, 8), (1024, 16)):
        n_tac = sum(p.numel() for p in TACExchange(dim).parameters())
        n_co = sum(p.numel() for p in BranchMHAExchange(dim, n_heads=heads).parameters())
        assert n_co - n_tac == 3 * dim - 3, (dim, heads, n_tac, n_co)
        assert n_tac > 4 * dim * dim, (dim, n_tac)  # 4 d^2 is the dominant term

    # At the real configuration the whole 22-block stack differs by 0.016% of
    # the model, so both arms print the same parameter count.
    depth, dim, model_params = 22, 1024, 429_441_750
    excess = depth * (3 * dim - 3)
    assert excess / model_params < 0.0002, excess

    # And the swap really reaches every injected block.
    coattn = build_tiny(
        ext_vocab_file,
        exchange={"type": "branch_mha", "schedule": BUILD_EXCHANGE["schedule"], "n_heads": 2},
    )
    tac = build_tiny(ext_vocab_file)
    assert {type(m.exchange).__name__ for m in coattn.modules() if isinstance(m, ExchangedBlock)} == {
        "BranchMHAExchange"
    }
    assert {type(m.exchange).__name__ for m in tac.modules() if isinstance(m, ExchangedBlock)} == {
        "TACExchange"
    }
