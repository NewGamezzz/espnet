"""Build BagPiper (speechlm-qwen3-8b) from a reconstructed train config and the
raw ESPnet safetensors shards, with strict two-way coverage checking.

Interface (imported by Task 6 and Task 12)::

    load_bagpiper(train_config_path, ckpt_path, device="cpu", dtype=torch.bfloat16)
        -> torch.nn.Module   # an eval-mode ParallelLLM (Qwen3ForCausalLM subclass)

The BagPiper deployment bundle ships the ESPnet state dict as 4 BF16 safetensors
shards with RAW ESPnet module keys (no renaming); there is no DeepSpeed .pt and no
train yaml (see docs/bagpiper-findings.md, Task 4). So the loader:

  * builds the model via SpeechLMJobTemplate.build_model() using the reconstructed
    config (which EXCLUDES the continuous_audio IO -- audio-input only, ~60 GB of
    Qwen3-Omni weights, not needed for TTS teacher-forcing);
  * merges the shards (``ckpt_path`` is the DIRECTORY holding them) -- or, for
    future use, loads a DeepSpeed-style .pt whose state dict lives under "module";
  * filters checkpoint tensors under ``multimodal_io_dict.continuous_audio.*`` and
    ``adaptor.continuous_audio.*`` (absent from the built model by construction);
  * asserts STRICT coverage BOTH ways on the retained model -- every built
    parameter gets a checkpoint tensor and every non-excluded checkpoint tensor
    lands in the model -- raising with the offending key lists otherwise.

NOTE on memory: the retained model is ~16.9 GB in bf16. On a 16 GiB machine the
build+forward does not fit in physical RAM (see docs/bagpiper-findings.md
"Gate results"); the strict-coverage CONTRACT is verified RAM-free by
``scripts/verify_key_coverage.py``.
"""

import glob
import os

import torch
import yaml

from espnet2.speechlm.model import _all_job_types

# Checkpoint key prefixes that the reconstructed (TTS-only) model does NOT build.
EXCLUDED_PREFIXES = (
    "multimodal_io_dict.continuous_audio.",
    "adaptor.continuous_audio.",
)


def load_raw_state_dict(ckpt_path):
    """Return the raw ESPnet state dict.

    ``ckpt_path`` is either a directory of ``model-*.safetensors`` shards (the
    tested BagPiper path) or a DeepSpeed ``.pt`` whose model state dict lives under
    the ``"module"`` key.
    """
    if os.path.isdir(ckpt_path):
        from safetensors.torch import load_file

        shards = sorted(glob.glob(os.path.join(ckpt_path, "model-*.safetensors")))
        if not shards:
            raise FileNotFoundError(
                f"No model-*.safetensors shards found under {ckpt_path!r}"
            )
        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(shard))
        return state_dict

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "module" in obj:
        return obj["module"]
    return obj


def filter_and_check_coverage(model_keys, raw_state_dict):
    """Filter excluded prefixes and assert strict two-way coverage.

    Returns ``(filtered_state_dict, excluded_keys)``. Raises ``RuntimeError`` with
    the offending key lists on any coverage gap.
    """
    filtered = {
        k: v for k, v in raw_state_dict.items() if not k.startswith(EXCLUDED_PREFIXES)
    }
    excluded_keys = [k for k in raw_state_dict if k.startswith(EXCLUDED_PREFIXES)]

    model_keys = set(model_keys)
    ckpt_keys = set(filtered.keys())
    missing = sorted(model_keys - ckpt_keys)     # built params with no ckpt tensor
    unexpected = sorted(ckpt_keys - model_keys)  # ckpt tensors with no built param
    if missing or unexpected:
        raise RuntimeError(
            "BagPiper strict-coverage load failed.\n"
            f"  {len(missing)} model params missing a checkpoint tensor: "
            f"{missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"  {len(unexpected)} checkpoint tensors with no model param: "
            f"{unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}\n"
            f"  ({len(excluded_keys)} continuous_audio/adaptor tensors were "
            "excluded by design.)"
        )
    return filtered, excluded_keys


def load_bagpiper(train_config_path, ckpt_path, device="cpu", dtype=torch.bfloat16):
    """Load BagPiper into an eval-mode module with strict-coverage checking."""
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

    job = _all_job_types[train_config["job_type"]](train_config, is_train=False)
    model = job.build_model()

    raw_sd = load_raw_state_dict(ckpt_path)
    filtered_sd, _ = filter_and_check_coverage(model.state_dict().keys(), raw_sd)

    # Coverage proven above; this is now a plain strict load.
    model.load_state_dict(filtered_sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()
