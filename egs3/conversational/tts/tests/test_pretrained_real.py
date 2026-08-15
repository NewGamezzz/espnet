"""Verification against the REAL pretrained F5TTS_Base checkpoint.

The unit suite proves the invariants on tiny random-init models; these tests
close the remaining gap by running the actual ``downloads/F5TTS_Base`` assets
through the production assembly path:

* vocab provenance and the appended-token id layout,
* the surgery loader (every backbone tensor bit-exact, embedding rows
  bit-exact, warm-started rows for the new tokens, gates zero after
  injection),
* single-channel ODE parity: with zero gates and ``counts=[1]`` the assembled
  multi-branch model must reproduce the baseline ``espnet2`` ``CFM`` output
  on identical inputs and seed (this simultaneously validates checkpoint
  loading, the surgery, injection identity, and the text-conditioning path),
* a forward-loss sanity check on real speech that catches architecture-config
  mistakes (``text_mask_padding`` / ``pe_attn_head``) which load cleanly but
  wreck the forward pass.

Skipped unless the checkpoint exists (download commands are in the header of
``conf/training_poc.yaml``).  CPU/fp32 only: parity needs bit-stable math and
``TACExchange`` is CUDA-nondeterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

from .conftest import REPO_ROOT  # noqa: F401  (side effect: repo root on sys.path)

from egs3.conversational.tts.dataset.preprocessing.text import (  # noqa: E402
    NEW_TOKENS,
    OTHER_TOKEN,
    TURN_TOKEN,
    extend_vocab,
    make_token2id,
    normalize_text,
    vocab_charset,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab  # noqa: E402
from egs3.conversational.tts.src.branch_exchange import ExchangedBlock  # noqa: E402
from egs3.conversational.tts.src.build_model import (  # noqa: E402
    build_multibranch_f5,
    check_vocab_provenance,
)
from espnet2.tts.f5.backbones.dit import DiT  # noqa: E402
from espnet2.tts.f5.cfm import CFM  # noqa: E402
from espnet2.tts.f5.inference import F5TTSInference  # noqa: E402

RECIPE_DIR = Path(__file__).resolve().parents[1]
CKPT = RECIPE_DIR / "downloads" / "F5TTS_Base" / "model_1200000.safetensors"
BASE_VOCAB = RECIPE_DIR / "downloads" / "F5TTS_Base" / "vocab.txt"
REF_WAV = RECIPE_DIR / "downloads" / "ref" / "basic_ref_en.wav"
# Transcript of REF_WAV (F5-TTS's own example reference clip).
REF_TEXT = "Some call me nature, others call me mother nature."

_TEXT_EMBED_KEY = "transformer.text_embed.text_embed.weight"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and BASE_VOCAB.exists()),
    reason="pretrained F5TTS_Base not downloaded (see conf/training_poc.yaml)",
)


def _poc_model_config() -> dict:
    """The ``model:`` block of conf/training_poc.yaml, interpolations resolved.

    Reading the real config (instead of copying values here) keeps these tests
    pinned to whatever the training run will actually build.
    """
    from omegaconf import OmegaConf

    config = OmegaConf.load(RECIPE_DIR / "conf" / "training_poc.yaml")
    config.recipe_dir = str(RECIPE_DIR)
    return OmegaConf.to_container(config, resolve=True)["model"]


@pytest.fixture(scope="module")
def model_config() -> dict:
    return _poc_model_config()


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    """Extended vocab + provenance meta built from the real base vocab,
    mirroring what the step-2 builder writes."""
    root = tmp_path_factory.mktemp("tokens")
    base = read_vocab(BASE_VOCAB)
    ext = extend_vocab(base)
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(ext) + "\n", encoding="utf-8")
    data = BASE_VOCAB.read_bytes()
    meta_file = root / "vocab_meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "base_vocab_sha256": hashlib.sha256(data).hexdigest(),
                "base_vocab_size": len(data.decode("utf-8").splitlines()),
            }
        ),
        encoding="utf-8",
    )
    return {"base": base, "ext": ext, "vocab_file": vocab_file, "meta_file": meta_file}


@pytest.fixture(scope="module")
def assembled(assets, model_config):
    """The full injected model exactly as conf/training_poc.yaml assembles it.

    Only the CFG train-drop probabilities are zeroed so the forward-loss test
    is deterministic; ``sample`` never reads them, so parity is unaffected.
    """
    cfm_config = dict(model_config["cfm"], audio_drop_prob=0.0, cond_drop_prob=0.0)
    model = build_multibranch_f5(
        vocab_file=str(assets["vocab_file"]),
        arch=model_config["arch"],
        cfm=cfm_config,
        exchange=model_config["exchange"],
        feats_extract=model_config["feats_extract"],
        pretrained_ckpt=str(CKPT),
        pretrained_vocab=str(BASE_VOCAB),
        vocab_meta=str(assets["meta_file"]),
        init_noise_scale=model_config["init_noise_scale"],
        init_seed=0,
    )
    return model.eval()


@pytest.fixture(scope="module")
def raw_state() -> dict:
    return F5TTSInference._load_native_f5_state(str(CKPT), use_ema=True)


def _encode(text: str, tokens: list[str]) -> torch.Tensor:
    """Token ids (1, nt) for ``text`` after vocab-charset normalization."""
    token2id = make_token2id(tokens)
    normalized = normalize_text(text, vocab_charset(tokens))
    return torch.tensor([[token2id[c] for c in normalized]], dtype=torch.long)


def test_vocab_provenance_accepts_real_vocab_and_rejects_mismatch(assets):
    check_vocab_provenance(assets["meta_file"], BASE_VOCAB)

    bad = json.loads(assets["meta_file"].read_text(encoding="utf-8"))
    bad["base_vocab_sha256"] = "0" * 64
    bad_file = assets["meta_file"].with_name("bad_meta.json")
    bad_file.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        check_vocab_provenance(bad_file, BASE_VOCAB)


def test_new_tokens_appended_after_real_vocab(assets):
    base_size = len(assets["base"])
    token2id = make_token2id(assets["ext"])
    assert assets["ext"][-len(NEW_TOKENS) :] == list(NEW_TOKENS)
    assert token2id[TURN_TOKEN] == base_size
    assert token2id[OTHER_TOKEN] == base_size + 1
    # The Emilia vocab carries a literal space token; the surgery warm-starts
    # <turn> from its row, so it must resolve.
    assert " " in token2id


def _wrapped_key(key: str, state_keys: set[str]) -> str:
    """Map a raw-checkpoint key to its post-injection name.

    ``inject_exchange`` replaces ``transformer_blocks[i]`` with an
    ``ExchangedBlock`` holding the original block as ``.base_block``.
    """
    if key in state_keys:
        return key
    wrapped = re.sub(
        r"^(transformer\.transformer_blocks\.\d+\.)", r"\1base_block.", key, count=1
    )
    assert wrapped in state_keys, f"no post-injection key for {key}"
    return wrapped


def test_assembled_weights_bitexact_vs_checkpoint(
    assembled, assets, raw_state, model_config
):
    """Every pretrained tensor must survive surgery + injection bit-exactly."""
    import torchaudio

    state = assembled.cfm.state_dict()
    state_keys = set(state)

    # The checkpoint's mel_spec.mel_stft.* keys are DSP constants of the
    # upstream torchaudio MelSpectrogram; the ported functional MelSpec has
    # no destination for them, so the loader drops them.  Verify here that
    # they equal a transform freshly built from the training config: a
    # mismatch would mean the checkpoint was trained with a different mel
    # front-end than this recipe uses.
    fe = model_config["feats_extract"]
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=fe["fs"],
        n_fft=fe["n_fft"],
        win_length=fe["win_length"],
        hop_length=fe["hop_length"],
        n_mels=fe["n_mels"],
        power=1,
        center=True,
        normalized=False,
        norm=None,
    )
    dsp_expected = {
        "mel_spec.mel_stft.spectrogram.window": mel_transform.spectrogram.window,
        "mel_spec.mel_stft.mel_scale.fb": mel_transform.mel_scale.fb,
    }

    matched = set()
    for key, ref in raw_state.items():
        if key.startswith("mel_spec.") and key not in state_keys:
            expected = dsp_expected.pop(key, None)
            assert expected is not None, f"unexpected dropped mel_spec key {key}"
            torch.testing.assert_close(ref, expected, rtol=0.0, atol=1e-6)
            continue
        mapped = _wrapped_key(key, state_keys)
        matched.add(mapped)
        if key == _TEXT_EMBED_KEY:
            new = state[mapped]
            base_rows = ref.shape[0]
            assert new.shape[0] == base_rows + len(NEW_TOKENS)
            assert torch.equal(new[:base_rows], ref), "original embedding rows changed"
            # Warm starts must differ from their sources (noise was added).
            assert not torch.equal(new[base_rows], ref[0])
            assert not torch.equal(new[base_rows + 1], ref[0])
        else:
            assert torch.equal(state[mapped], ref), f"weight mismatch at {key}"

    # Whatever the checkpoint did not provide must be exchange-only.
    leftover = state_keys - matched
    assert leftover, "injection added no parameters?"
    assert all(".exchange." in key for key in leftover), sorted(leftover)[:5]

    # Zero-init gates: at this instant the model is N independent F5 passes.
    gates = [
        module.exchange.g
        for module in assembled.modules()
        if isinstance(module, ExchangedBlock)
    ]
    assert len(gates) == len(assembled.cfm.transformer.transformer_blocks)
    assert all(torch.all(gate == 0) for gate in gates)


def test_single_channel_sample_parity_vs_baseline_cfm(
    assembled, assets, raw_state, model_config
):
    """The gold check: counts=[1] + zero gates == baseline espnet2 CFM.

    Bit-identical inputs and seed must give (numerically) identical mels.
    A pass proves checkpoint loading, embedding surgery, injection identity
    and text conditioning in one shot.  ~2 min on CPU at steps=4.
    """
    arch = model_config["arch"]
    fe = model_config["feats_extract"]
    baseline = CFM(
        DiT(mel_dim=fe["n_mels"], text_num_embeds=len(assets["base"]), **arch),
        num_channels=fe["n_mels"],
        odeint_kwargs=dict(method="euler"),
        mel_spec_kwargs=dict(
            n_fft=fe["n_fft"],
            hop_length=fe["hop_length"],
            win_length=fe["win_length"],
            n_mel_channels=fe["n_mels"],
            target_sample_rate=fe["fs"],
            mel_spec_type=fe["mel_spec_type"],
        ),
    )
    # Same mel_spec DSP-buffer situation as the surgery loader: the ported
    # functional MelSpec has no destination for the checkpoint's
    # mel_spec.mel_stft.* constants (validated in the bit-exact test above).
    baseline_keys = set(baseline.state_dict())
    baseline.load_state_dict(
        {
            k: v
            for k, v in raw_state.items()
            if not (k.startswith("mel_spec.") and k not in baseline_keys)
        },
        strict=True,
    )
    baseline.eval()

    # Base-vocab-only text: identical ids resolve to identical (bit-copied)
    # embedding rows in both models.
    ids = _encode(REF_TEXT, assets["base"])

    generator = torch.Generator().manual_seed(123)
    cond = torch.randn(1, 96, fe["n_mels"], generator=generator)
    lens = torch.tensor([96], dtype=torch.long)
    sample_kwargs = dict(
        duration=160,
        lens=lens,
        steps=4,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        seed=7,
    )

    # Seed handling matches by construction at batch size 1: baseline reseeds
    # before its (single) noise draw; MultiBranchCFM seeds once up front.
    with torch.inference_mode():
        out_baseline, _ = baseline.sample(cond, ids, **sample_kwargs)
        out_multi, _ = assembled.cfm.sample(cond, ids, counts=[1], **sample_kwargs)

    torch.testing.assert_close(out_multi, out_baseline, rtol=0.0, atol=1e-5)


@pytest.mark.skipif(not REF_WAV.exists(), reason="reference clip not downloaded")
def test_pretrained_loss_beats_zero_predictor_on_real_speech(assembled, assets):
    """Forward flow-matching loss on real speech must land far below the
    zero-predictor bound.

    DiT's init zeroes ``proj_out``, so an UNLOADED model scores exactly that
    bound; a loaded model that only looks loaded (e.g. ``text_mask_padding``
    or ``pe_attn_head`` wrong: same parameter names, different forward) also
    stays near it.  Real pretrained weights must beat it decisively.
    """
    import soundfile as sf

    data, sr = sf.read(str(REF_WAV), dtype="float32")
    assert sr == 24000, f"reference clip must be 24 kHz, got {sr}"
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, T_wav) mono

    mel = assembled.cfm.mel_spec(wav).permute(0, 2, 1)  # (1, T, n_mels)
    ids = _encode(REF_TEXT, assets["ext"])

    x0 = torch.randn(
        mel.shape, generator=torch.Generator().manual_seed(0), dtype=mel.dtype
    )
    with torch.no_grad():
        loss, stats, extras = assembled.cfm(
            mel,
            ids,
            counts=[1],
            frac_lengths=torch.tensor([0.8]),
            time=torch.tensor([0.5]),
            x0=x0,
        )

    span = extras["rand_span_mask"]
    zero_predictor = ((mel - x0) ** 2)[span].mean()
    assert torch.isfinite(loss)
    assert loss < 0.5 * zero_predictor, (
        f"pretrained loss {loss.item():.4f} not clearly below the "
        f"zero-predictor bound {zero_predictor.item():.4f}; the weights or "
        "the architecture config are wrong"
    )
