"""RAM-free proof of the load_bagpiper strict-coverage contract (names + shapes).

The retained BagPiper model is ~16.9 GB in bf16 and does not fit in this machine's
16 GiB RAM, so we cannot call ``job.build_model()`` (which loads the full
Qwen3-8B-Base base weights) here. Instead we reconstruct the EXACT set of
(key, shape) pairs that ``build_model()`` would produce, deriving every size from
the *reconstructed config itself* (not hardcoded constants), without materializing
any large weights:

  * instantiate ``SpeechLMJobTemplate`` (builds only the small IOs + the vocab,
    NOT the 16 GB model) -> ``job.vocab`` / ``job.vocab_intervals`` /
    ``discrete_audio.num_stream()`` drive vocab_size and num_stream, so a bad
    config (wrong codec_max_token_per_frame, wrong tokenizer, ...) is caught here;
  * backbone: meta-instantiate ``transformers.Qwen3ForCausalLM`` with
    ``vocab_size`` from the job -- ParallelLLM subclasses this and only replaces
    embed_tokens/lm_head with same-named modules of that vocab + adds stream_emb;
  * ``stream_emb.weight`` -> ``[num_stream, hidden]`` from the job;
  * ``discrete_audio``: reuse the job's REAL (small) DiscreteAudioIO state dict,
    prefixed ``multimodal_io_dict.discrete_audio.``;
  * ``text``: the job's REAL HuggingFaceTextIO (expected to contribute no keys);
  * ``adaptor``: empty (no continuous IO in the reconstructed config).

We then compare that expected {key: shape} map against the checkpoint's own
tensors (names + shapes from the safetensors headers) minus the excluded
continuous_audio/adaptor keys, asserting equality both ways. Shapes matter:
``load_state_dict`` raises on a shape mismatch independent of ``strict``, so a
wrong ``num_stream`` would give an identically-named ``stream_emb.weight`` of the
wrong shape -- names alone would not catch it.
"""

import json
import os
import struct
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_bagpiper import EXCLUDED_PREFIXES  # noqa: E402

CONF = os.path.join(HERE, "..", "conf", "bagpiper_train_config.yaml")
CKPT_DIR = os.path.join(HERE, "..", "downloads", "bagpiper", "speechlm-qwen3-8b")
INDEX = os.path.join(CKPT_DIR, "model.safetensors.index.json")

# Ground-truth sizes from downloads/.../config.json (validation targets).
EXPECT_VOCAB = 160392
EXPECT_NUM_STREAM = 8


def checkpoint_shapes():
    """{key: tuple(shape)} for every checkpoint tensor, from safetensors headers."""
    idx = json.load(open(INDEX))
    shapes = {}
    for shard in set(idx["weight_map"].values()):
        path = os.path.join(CKPT_DIR, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            shapes[k] = tuple(v["shape"])
    return shapes


def expected_shapes(config):
    import transformers

    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

    # Job build is RAM-cheap: it constructs the small IOs + the unified vocab,
    # NOT the 16 GB transformer. Every size below flows from the config.
    job = SpeechLMJobTemplate(config, is_train=False)
    vocab_size = max(e for iv in job.vocab_intervals.values() for _, e in iv)
    assert len(job.vocab) == vocab_size == EXPECT_VOCAB, (
        f"reconstructed vocab {len(job.vocab)} != config.json {EXPECT_VOCAB}"
    )
    assert job.vocab_intervals["text"][0] == (256, 152192), job.vocab_intervals["text"]
    assert job.vocab_intervals["discrete_audio"][0][0] == 152192

    da = job.multimodal_io["discrete_audio"]
    num_stream = da.num_stream()
    assert num_stream == EXPECT_NUM_STREAM, f"num_stream {num_stream} != 8"

    model_conf = config["model"]
    cfg = transformers.AutoConfig.from_pretrained(model_conf["model_hf_tag"])
    arch = getattr(transformers, cfg.architectures[0])
    hidden = cfg.hidden_size
    cfg.vocab_size = vocab_size
    cfg.tie_word_embeddings = False
    with torch.device("meta"):
        backbone = arch(cfg)

    shapes = {k: tuple(v.shape) for k, v in backbone.state_dict().items()}
    shapes["stream_emb.weight"] = (num_stream, hidden)
    for k, v in da.state_dict().items():
        shapes[f"multimodal_io_dict.discrete_audio.{k}"] = tuple(v.shape)
    txt = job.multimodal_io["text"]
    n_txt = 0
    for k, v in txt.state_dict().items():
        shapes[f"multimodal_io_dict.text.{k}"] = tuple(v.shape)
        n_txt += 1
    return shapes, arch.__name__, len(da.state_dict()), n_txt, vocab_size, num_stream


def main():
    with open(CONF) as f:
        config = yaml.safe_load(f)

    ckpt = checkpoint_shapes()
    excluded = {k for k in ckpt if k.startswith(EXCLUDED_PREFIXES)}
    retained = {k: ckpt[k] for k in ckpt if k not in excluded}

    expected, arch_name, n_da, n_txt, vocab_size, num_stream = expected_shapes(config)

    missing = sorted(set(expected) - set(retained))     # built, no ckpt tensor
    unexpected = sorted(set(retained) - set(expected))  # ckpt tensor, no built param
    shape_mismatch = sorted(
        k for k in (set(expected) & set(retained)) if expected[k] != retained[k]
    )

    print(f"backbone class            : {arch_name}")
    print(f"reconstructed vocab_size  : {vocab_size}   num_stream: {num_stream}")
    print(f"checkpoint tensors total  : {len(ckpt)}")
    print(f"  excluded (continuous)   : {len(excluded)}")
    print(f"  retained (must load)    : {len(retained)}")
    print(f"expected built-model keys : {len(expected)}")
    print(f"  discrete_audio IO keys  : {n_da}")
    print(f"  text IO keys            : {n_txt}")
    print(f"missing (model w/o ckpt)  : {len(missing)}")
    print(f"unexpected (ckpt w/o model): {len(unexpected)}")
    print(f"shape mismatches          : {len(shape_mismatch)}")
    if missing:
        print("  MISSING:", missing[:30])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:30])
    for k in shape_mismatch[:30]:
        print(f"  SHAPE {k}: expected {expected[k]} vs ckpt {retained[k]}")

    ok = not missing and not unexpected and not shape_mismatch
    print(
        "\nRESULT:",
        "PASS - strict two-way coverage holds (names + shapes)" if ok else "FAIL",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
