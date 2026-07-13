"""RAM-free proof of the load_bagpiper strict-coverage contract.

The retained BagPiper model is ~16.9 GB in bf16 and does not fit in this machine's
16 GiB RAM, so we cannot call ``job.build_model()`` (which would load the full
Qwen3-8B-Base base weights) here. Instead we reconstruct the EXACT key set that
``build_model()`` would produce, without materializing any large weights:

  * backbone: meta-instantiate ``transformers.Qwen3ForCausalLM`` with the enlarged
    (vocab_size=160392) config -- ParallelLLM subclasses this class and only
    replaces embed_tokens/lm_head with same-named modules + adds stream_emb;
  * stream_emb.weight: added by ParallelLLM.from_pretrained;
  * discrete_audio: instantiate the REAL (small) DiscreteAudioIO and prefix its
    state_dict keys with ``multimodal_io_dict.discrete_audio.``;
  * text: instantiate the REAL HuggingFaceTextIO (expected to contribute no keys);
  * adaptor: empty (no continuous IO in the reconstructed config).

We then compare that expected set against the checkpoint's own key list (from
model.safetensors.index.json) minus the excluded continuous_audio/adaptor keys,
asserting equality both ways -- the same contract load_bagpiper enforces at load
time. If this passes, Task 6 / Task 12 can import load_bagpiper and get correct,
complete coverage on any machine with enough RAM.
"""

import json
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE))

from load_bagpiper import EXCLUDED_PREFIXES  # noqa: E402

CONF = os.path.join(HERE, "..", "conf", "bagpiper_train_config.yaml")
CKPT_DIR = os.path.join(HERE, "..", "downloads", "bagpiper", "speechlm-qwen3-8b")
INDEX = os.path.join(CKPT_DIR, "model.safetensors.index.json")


def checkpoint_keys():
    idx = json.load(open(INDEX))
    return set(idx["weight_map"].keys())


def expected_model_keys(config):
    import transformers

    from espnet2.speechlm.model.speechlm.multimodal_io.audio import DiscreteAudioIO
    from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO

    io_conf = config["multimodal_io"]
    model_conf = config["model"]

    # (1) backbone -- meta init, no weights materialized.
    cfg = transformers.AutoConfig.from_pretrained(model_conf["model_hf_tag"])
    arch = getattr(transformers, cfg.architectures[0])
    cfg.vocab_size = 160392  # rebuilt multimodal vocab; embed/lm_head resized in place
    cfg.tie_word_embeddings = False
    with torch.device("meta"):
        backbone = arch(cfg)
    keys = set(backbone.state_dict().keys())

    # (2) stream_emb added by ParallelLLM.from_pretrained.
    keys.add("stream_emb.weight")

    # (3) discrete_audio IO -- real, small.
    da = DiscreteAudioIO(**io_conf["discrete_audio"])
    da_keys = set(da.state_dict().keys())
    keys |= {f"multimodal_io_dict.discrete_audio.{k}" for k in da_keys}

    # (4) text IO -- expected to contribute no persistent keys.
    txt = HuggingFaceTextIO(**io_conf["text"])
    txt_keys = set(txt.state_dict().keys())
    keys |= {f"multimodal_io_dict.text.{k}" for k in txt_keys}

    return keys, arch.__name__, len(da_keys), len(txt_keys)


def main():
    with open(CONF) as f:
        config = yaml.safe_load(f)

    ckpt = checkpoint_keys()
    excluded = {k for k in ckpt if k.startswith(EXCLUDED_PREFIXES)}
    retained = ckpt - excluded

    expected, arch_name, n_da, n_txt = expected_model_keys(config)

    missing = sorted(expected - retained)     # built params with no ckpt tensor
    unexpected = sorted(retained - expected)  # ckpt tensors with no built param

    print(f"backbone class            : {arch_name}")
    print(f"checkpoint tensors total  : {len(ckpt)}")
    print(f"  excluded (continuous)   : {len(excluded)}")
    print(f"  retained (must load)    : {len(retained)}")
    print(f"expected built-model keys : {len(expected)}")
    print(f"  discrete_audio IO keys  : {n_da}")
    print(f"  text IO keys            : {n_txt}")
    print(f"missing (model w/o ckpt)  : {len(missing)}")
    print(f"unexpected (ckpt w/o model): {len(unexpected)}")
    if missing:
        print("  MISSING:", missing[:30])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:30])

    ok = not missing and not unexpected
    print("\nRESULT:", "PASS - strict two-way coverage holds" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
