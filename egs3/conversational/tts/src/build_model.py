"""Assembly of the multi-branch F5 POC model (order is load-bearing).

``build_multibranch_f5`` is the single entry point the trainer instantiates
(``model._target_`` in ``conf/training_poc.yaml``):

1. Build the DiT/CFM with the F5TTS_Base architecture values and
   ``text_num_embeds`` = size of the EXTENDED vocab from step 2.
2. Load the pretrained F5TTS_Base checkpoint.  The text-embedding weight
   mismatches in shape by exactly the four appended tokens; every original
   row is copied bit-exactly and the four new rows are warm-started
   (``<turn>`` from the space character's row; ``<OTHER>``,
   ``<speaker_prompt>``, and ``<prev_chunk>`` from the filler row 0 - F5's
   internal padding token, the closest pretrained concept to "no text for
   me here" - each plus small Gaussian noise).  Everything
   else must load exactly (strict load, zero missing/unexpected keys),
   except the checkpoint's ``mel_spec.mel_stft.*`` DSP buffers, which the
   ported functional MelSpec does not register and which are dropped after
   verification-by-test (see ``load_pretrained_with_surgery``).
3. ``inject_exchange`` with the configured schedule.  Gates are zero-init,
   so at this instant the model computes exactly N independent pretrained
   F5 passes.
4. The optimizer's two param groups (exchange vs backbone) are built by
   ``exchange_param_groups`` at ``configure_optimizers`` time (see
   ``src/lit_module.py``).

Steps 2 and 3 must run in this order: injection wraps blocks in
``ExchangedBlock`` and would shift the backbone's state-dict keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from egs3.conversational.tts.dataset.preprocessing.text import (
    NEW_TOKENS,
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab
from egs3.conversational.tts.src.branch_exchange import (
    REGISTRY,
    BranchContext,
    BranchMHAExchange,
    ExchangedBlock,
    ExchangeSchedule,
    IdentityExchange,
    TACExchange,
    inject_exchange,
)
from egs3.conversational.tts.src.model import MultiBranchF5
from egs3.conversational.tts.src.multibranch_cfm import MultiBranchCFM
from espnet2.tts.f5.backbones.dit import DiT
from espnet2.tts.feats_extract.vocoder_mel import VocoderMelSpec

_TEXT_EMBED_KEY = "transformer.text_embed.text_embed.weight"


def _as_dict(config) -> dict:
    if config is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    return dict(config)


def extended_text_embedding(
    pretrained_weight: torch.Tensor,
    tokens: list[str],
    noise_scale: float = 0.02,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Warm-started embedding matrix for the extended vocab.

    ``pretrained_weight`` is the F5TTS_Base ``text_embed`` matrix of shape
    ``(base_size + 1, text_dim)`` (row 0 is the filler token; token id i
    lives in row i+1).  ``tokens`` is the extended vocab whose last
    ``len(NEW_TOKENS)`` entries must be ``NEW_TOKENS`` (step 2 appends them
    at the end, so all original ids - and therefore rows - are unchanged).
    Five appended tokens are warm-started: ``<turn>`` from the space
    character's row; ``<OTHER>``, ``<speaker_prompt>``, ``<prev_chunk>``,
    and ``<turn_fill>`` from the filler row 0 - F5's learned "audio beyond
    text" representation - each plus small Gaussian noise.
    """
    if list(tokens[-len(NEW_TOKENS) :]) != list(NEW_TOKENS):
        raise ValueError(
            f"extended vocab must end with {NEW_TOKENS}, got "
            f"{tokens[-len(NEW_TOKENS):]!r}"
        )
    base_size = len(tokens) - len(NEW_TOKENS)
    if pretrained_weight.shape[0] != base_size + 1:
        raise ValueError(
            f"pretrained embedding has {pretrained_weight.shape[0]} rows, "
            f"expected base vocab {base_size} + 1 filler; the checkpoint and "
            "the extended vocab do not belong together"
        )
    token2id = make_token2id(list(tokens))
    space_row = token2id[" "] + 1
    turn_row = token2id[TURN_TOKEN] + 1  # == base_size + 1
    other_row = token2id[OTHER_TOKEN] + 1  # == base_size + 2
    sp_row = token2id[SPEAKER_PROMPT_TOKEN] + 1  # == base_size + 3
    pc_row = token2id[PREV_CHUNK_TOKEN] + 1  # == base_size + 4
    tf_row = token2id[TURN_FILL_TOKEN] + 1  # == base_size + 5

    def _noise() -> torch.Tensor:
        return noise_scale * torch.randn(
            pretrained_weight.shape[1],
            generator=generator,
            dtype=pretrained_weight.dtype,
        )

    new_weight = pretrained_weight.new_empty(
        (len(tokens) + 1, pretrained_weight.shape[1])
    )
    new_weight[: base_size + 1] = pretrained_weight
    new_weight[turn_row] = pretrained_weight[space_row] + _noise()
    new_weight[other_row] = pretrained_weight[0] + _noise()
    # Audio-only conditioning spans: warm-start from the filler row, F5's
    # learned "audio beyond text" representation (design decision 2026-08-14).
    new_weight[sp_row] = pretrained_weight[0] + _noise()
    new_weight[pc_row] = pretrained_weight[0] + _noise()
    new_weight[tf_row] = pretrained_weight[0] + _noise()
    return new_weight


def check_vocab_provenance(vocab_meta: str | Path, pretrained_vocab: str | Path):
    """Assert step 2's recorded base vocab matches the checkpoint's vocab file.

    ``vocab_meta`` is the builder's ``tokens/vocab_meta.json``;
    ``pretrained_vocab`` is the vocab file shipped with the pretrained
    checkpoint.  A mismatch means the extended vocab was built from a
    different base than the checkpoint expects - ids would silently point
    at wrong embedding rows.
    """
    meta = json.loads(Path(vocab_meta).read_text(encoding="utf-8"))
    data = Path(pretrained_vocab).read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data.decode("utf-8").splitlines())
    if sha256 != meta["base_vocab_sha256"] or size != meta["base_vocab_size"]:
        raise ValueError(
            f"pretrained vocab {pretrained_vocab} (sha256 {sha256}, "
            f"{size} tokens) does not match vocab_meta.json "
            f"(sha256 {meta['base_vocab_sha256']}, "
            f"{meta['base_vocab_size']} tokens)"
        )


def load_pretrained_with_surgery(
    cfm: MultiBranchCFM,
    pretrained_ckpt: str | Path,
    tokens: list[str],
    noise_scale: float = 0.02,
    generator: torch.Generator | None = None,
) -> None:
    """Load a native F5-TTS checkpoint into ``cfm`` with embedding surgery."""
    # Reuse the espnet2 inference loader for the official checkpoint formats
    # (.pt / .safetensors, EMA-prefixed keys, bookkeeping tensors dropped).
    from espnet2.tts.f5.inference import F5TTSInference

    state = F5TTSInference._load_native_f5_state(str(pretrained_ckpt), use_ema=True)
    if _TEXT_EMBED_KEY not in state:
        raise KeyError(f"{pretrained_ckpt} has no {_TEXT_EMBED_KEY}")
    state[_TEXT_EMBED_KEY] = extended_text_embedding(
        state[_TEXT_EMBED_KEY], tokens, noise_scale=noise_scale, generator=generator
    )
    # The official checkpoint EMA-tracked the upstream MelSpec's torchaudio
    # submodule buffers (mel_spec.mel_stft.*: hann window + mel filterbank).
    # The ported MelSpec computes the mel functionally and registers no such
    # buffers, so these keys have no destination.  They are DSP constants
    # derived from the mel config, not weights (tests/test_pretrained_real.py
    # verifies them against a freshly built transform), so drop exactly the
    # mel_spec keys the model does not carry and keep everything else strict.
    model_keys = set(cfm.state_dict())
    for key in [k for k in state if k.startswith("mel_spec.") and k not in model_keys]:
        del state[key]
    # strict: after the surgery every remaining key must match exactly.
    cfm.load_state_dict(state, strict=True)


def _exchange_factory(exchange_config: dict, dim: int):
    config = dict(exchange_config)
    kind = config.pop("type")
    config.pop("schedule", None)
    if kind == "tac":
        return lambda: TACExchange(dim, **config)
    if kind == "branch_mha":
        return lambda: BranchMHAExchange(dim, **config)
    if kind == "identity":
        return lambda: IdentityExchange()
    raise ValueError(f"unknown exchange type {kind!r}")


def exchange_param_groups(
    model: torch.nn.Module, lr_exchange: float, lr_backbone: float
) -> list[dict]:
    """Two optimizer param groups: injected exchange modules vs everything else."""
    exchange_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, ExchangedBlock)
        for p in m.exchange.parameters()
    }
    if not exchange_ids:
        raise ValueError("model has no injected exchanges; inject_exchange first")
    exchange_params, backbone_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (exchange_params if id(p) in exchange_ids else backbone_params).append(p)
    return [
        {"params": exchange_params, "lr": lr_exchange},
        {"params": backbone_params, "lr": lr_backbone},
    ]


def build_multibranch_f5(
    vocab_file: str,
    arch=None,
    cfm=None,
    exchange=None,
    feats_extract=None,
    pretrained_ckpt: str | None = None,
    pretrained_vocab: str | None = None,
    vocab_meta: str | None = None,
    init_noise_scale: float = 0.02,
    init_seed: int = 0,
) -> MultiBranchF5:
    """Build the injected multi-branch F5 model (see module docstring).

    Args:
        vocab_file: The EXTENDED vocab from step 2 (line index = token id).
        arch: DiT kwargs (copy the F5TTS_Base values verbatim).
        cfm: CFM kwargs (sigma, drop probs, frac_lengths_mask, odeint_method).
        exchange: ``{type, schedule, **module kwargs}``; schedule keys are
            1-indexed inclusive ranges, e.g. ``{"1-22": "P+TAC"}``.
        feats_extract: ``VocoderMelSpec`` kwargs (fs/n_fft/hop/win/n_mels).
        pretrained_ckpt: Native F5-TTS checkpoint (.pt/.safetensors);
            ``None`` keeps the random init (tests only).
        pretrained_vocab: Vocab file shipped with the checkpoint; checked
            against ``vocab_meta`` before any weight is loaded.
        vocab_meta: Step 2's ``tokens/vocab_meta.json``.
    """
    arch = _as_dict(arch)
    cfm_config = _as_dict(cfm)
    exchange_config = _as_dict(exchange) or {
        "type": "tac",
        "schedule": {"1-22": "P+TAC"},
    }
    feats_config = _as_dict(feats_extract)

    tokens = read_vocab(vocab_file)
    if list(tokens[-len(NEW_TOKENS) :]) != list(NEW_TOKENS):
        raise ValueError(
            f"{vocab_file} must be the step-2 extended vocab ending with "
            f"{NEW_TOKENS}, got {tokens[-len(NEW_TOKENS):]!r}"
        )

    if pretrained_ckpt is not None:
        if vocab_meta is None or pretrained_vocab is None:
            raise ValueError(
                "pretrained_ckpt requires vocab_meta and pretrained_vocab so "
                "the base-vocab provenance can be verified before loading"
            )
        check_vocab_provenance(vocab_meta, pretrained_vocab)

    extractor = VocoderMelSpec(**feats_config)
    mel_dim = extractor.output_size()

    # 1. DiT + CFM with the extended vocab size.
    transformer = DiT(
        mel_dim=mel_dim,
        text_num_embeds=len(tokens),
        **arch,
    )
    odeint_method = cfm_config.pop("odeint_method", "euler")
    ctx = BranchContext()
    # The CFM's internal MelSpec (used by CFM.sample on raw-wave prompts, and
    # by forward's raw-wave branch) must stay bit-compatible with the
    # training-time feats_extract; without this the two front-ends silently
    # diverge as soon as the feats_extract block changes (degraded audio,
    # never an error). Note the key renames vs VocoderMelSpec.
    mel_spec_kwargs = dict(
        n_fft=extractor.n_fft,
        hop_length=extractor.hop_length,
        win_length=extractor.win_length,
        n_mel_channels=extractor.n_mels,
        target_sample_rate=extractor.fs,
        mel_spec_type=extractor.mel_spec_type,
    )
    model_cfm = MultiBranchCFM(
        transformer,
        ctx=ctx,
        num_channels=mel_dim,
        odeint_kwargs=dict(method=odeint_method),
        mel_spec_kwargs=mel_spec_kwargs,
        **cfm_config,
    )

    # 2. Pretrained weights + embedding surgery (before injection: wrapping
    #    would shift the backbone state-dict keys).
    if pretrained_ckpt is not None:
        generator = torch.Generator().manual_seed(init_seed)
        load_pretrained_with_surgery(
            model_cfm,
            pretrained_ckpt,
            tokens,
            noise_scale=init_noise_scale,
            generator=generator,
        )

    # 3. Inject the exchanges (zero-init gates: exact identity at this point).
    schedule = ExchangeSchedule.from_spec(
        _as_dict(exchange_config.get("schedule")) or {"1-22": "P+TAC"},
        depth=transformer.depth,
        factory=_exchange_factory(exchange_config, transformer.dim),
    )
    inject_exchange(transformer, REGISTRY["f5_dit"], schedule, ctx)

    return MultiBranchF5(cfm=model_cfm, feats_extract=extractor)
