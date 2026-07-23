"""Model factory for multi-compression codec fine-tuning.

Referenced from the training config as a Hydra ``_target_`` (the config
must NOT set a top-level ``task:`` key, so ``CodecSystem`` falls back to
``instantiate(config.model)``):

.. code-block:: yaml

    model:
      _target_: src.factory.build_multicomp_model
      pretrained_train_config: /path/to/baseline_exp/config.yaml
      pretrained_model_file: /path/to/baseline_exp/last.ckpt
      compression_model:
        name: cosine_similarity
        params: {}
        min_rate: 0.2
        max_rate: 1.0
        random_rate: per_quantizer
        eval_rate: 0.5
      freeze_codec_module: none

The factory is the recipe's ONLY weight-loading mechanism (deliberately
recipe-local: no espnet3 framework hook is involved, so other recipes'
loading conventions are unaffected).  ``pretrained_model_file`` is
strict-loaded into the UNWRAPPED base model BEFORE the quantizer is
wrapped, so the checkpoint's natural key layout matches and nothing can
overwrite the weights afterwards - on the task-less Hydra path nothing
else touches the model between ``instantiate(config.model)`` and
``fit``.  ``pretrained_model_tag`` (HF zoo) inherently carries its own
weights instead.  The weight path is excluded from the ``dump_config_to``
spec, so the inference-time rebuild never depends on the baseline
checkpoint still existing.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from omegaconf import OmegaConf

from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

from .compression_models import build_compression_model
from .wrappers import CompressionResidualVectorQuantizer

FREEZE_CHOICES = ("none", "encoder", "decoder")


def _to_plain(config):
    """Convert possible OmegaConf containers from Hydra into plain dicts."""
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _build_base_model(
    codec: Optional[str],
    codec_conf: Optional[Dict[str, Any]],
    task: str,
    pretrained_train_config: Optional[str],
    pretrained_model_tag: Optional[str],
) -> AbsGANESPnetModel:
    """Build the unwrapped base codec model (architecture only)."""
    sources = [
        pretrained_model_tag is not None,
        pretrained_train_config is not None,
        codec is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Specify exactly one model source: 'pretrained_model_tag' (HF zoo), "
            "'pretrained_train_config' (espnet2-style config.yaml of a previous "
            "run), or 'codec'+'codec_conf' (fresh config)."
        )

    if pretrained_model_tag is not None:
        # Zoo download; the tag inherently carries its own weights.
        from espnet2.bin.gan_codec_inference import AudioCoding

        return AudioCoding.from_pretrained(model_tag=pretrained_model_tag).model

    from espnet3.utils.task_utils import get_espnet_model

    if pretrained_train_config is not None:
        # Build from the previous run's espnet2-style config.yaml. Its
        # weights are loaded separately via pretrained_model_file (or
        # the fine-tuned checkpoint at inference). Merging over the task
        # defaults also tolerates partial configs.
        with open(pretrained_train_config, encoding="utf-8") as f:
            train_args = yaml.safe_load(f)
        return get_espnet_model(task, train_args)

    return get_espnet_model(
        task, {"codec": codec, "codec_conf": _to_plain(codec_conf) or {}}
    )


def load_model_state_strict(model: torch.nn.Module, model_file: str) -> None:
    """Strictly load a model-level or Lightning checkpoint.

    Lightning checkpoints (``last.ckpt``) nest the weights under a
    ``state_dict`` key; they are unwrapped here (and a legacy ``model.``
    key prefix is stripped when present).  Unlike espnet2's own
    ``strict=False`` loading - which silently yields a random model when
    keys do not match (e.g. the framework's EMPTY ``ave_Nbest.pth``
    files) - the load is always strict, so any mismatch raises.
    """
    state_dict = torch.load(model_file, map_location="cpu")
    if "state_dict" in state_dict:  # Lightning checkpoint layout
        state_dict = state_dict["state_dict"]
    if any(k.startswith("model.") for k in state_dict):
        # Older ESPnetLightningModule versions prefixed the inner model.
        state_dict = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
    model.load_state_dict(state_dict, strict=True)


def freeze_codec_module(model: AbsGANESPnetModel, module: str = "none") -> None:
    """Freeze parts of the codec generator before fine-tuning.

    Only ``none``/``encoder``/``decoder`` are supported: freezing both (or
    ``all``) would leave the recipe's named ``generator`` optimizer without
    any trainable parameters (the EMA codebooks are buffers, not
    parameters), which the optimizer validation rejects.
    """
    if module not in FREEZE_CHOICES:
        raise ValueError(
            f"freeze_codec_module must be one of {FREEZE_CHOICES}, got '{module}'."
        )
    if module == "none":
        return
    for param in getattr(model.codec.generator, module).parameters():
        param.requires_grad = False


def build_multicomp_model(
    compression_model: Dict[str, Any],
    codec: Optional[str] = None,
    codec_conf: Optional[Dict[str, Any]] = None,
    task: str = "espnet2.tasks.gan_codec.GANCodecTask",
    pretrained_train_config: Optional[str] = None,
    pretrained_model_file: Optional[str] = None,
    pretrained_model_tag: Optional[str] = None,
    freeze_codec_module_name: str = "none",
    dump_config_to: Optional[str] = None,
) -> AbsGANESPnetModel:
    """Build a codec wrapped with the multi-compression quantizer.

    ``pretrained_model_file`` (when given) is strict-loaded into the
    UNWRAPPED base model before wrapping - the factory is the recipe's
    only weight-loading mechanism.  Inference instead rebuilds the
    architecture from the dumped spec (which never contains the weight
    path) and loads the fine-tuned checkpoint on top.

    Args:
        compression_model: dict with ``name`` (registry key), optional
            ``params`` for the compression model itself, and the wrapper's
            rate settings (``min_rate``, ``max_rate``, ``random_rate``,
            ``eval_rate``).
        codec / codec_conf: fresh GANCodecTask model spec (mutually
            exclusive with the pretrained_* sources).
        task: espnet2 task path used with ``codec``/``codec_conf``.
        pretrained_train_config: espnet2-style config.yaml of a previous
            run (built via ``save_espnet_config``); architecture only.
        pretrained_model_file: baseline checkpoint strict-loaded into the
            unwrapped model before wrapping - either a model-level state
            dict or a Lightning checkpoint (``last.ckpt``), which is
            unwrapped automatically.  Not allowed together with
            ``pretrained_model_tag`` (the tag carries its own weights).
        pretrained_model_tag: espnet_model_zoo tag (e.g.
            ``espnet/libritts_encodec_24k``); weights come with the tag.
        freeze_codec_module_name: ``none`` | ``encoder`` | ``decoder``.
        dump_config_to: when set (training config points it at
            ``${exp_dir}/multicomp_model.yaml``), the resolved model spec
            (architecture only, no ``pretrained_model_file``) is written
            there so inference can rebuild the wrapped architecture and
            load the fine-tuned checkpoint on top.  This substitutes for
            ``save_espnet_config``, which ``CodecSystem.train`` skips on
            the task-less Hydra path.

    Returns:
        ``ESPnetGANCodecModel`` (an ``AbsGANESPnetModel``, so
        ``CodecSystem`` selects the GAN trainer automatically).
    """
    if pretrained_model_file is not None and pretrained_model_tag is not None:
        raise ValueError(
            "'pretrained_model_file' cannot be combined with "
            "'pretrained_model_tag': the zoo download already carries its "
            "own weights."
        )

    model = _build_base_model(
        codec=codec,
        codec_conf=codec_conf,
        task=task,
        pretrained_train_config=pretrained_train_config,
        pretrained_model_tag=pretrained_model_tag,
    )

    # Load BEFORE wrapping: the checkpoint's natural (unwrapped) key layout
    # matches the model exactly, so the load can stay strict.
    if pretrained_model_file is not None:
        load_model_state_strict(model, pretrained_model_file)

    quantizer = model.codec.generator.quantizer
    if not isinstance(quantizer, ResidualVectorQuantizer):
        raise TypeError(
            "Multi-compression wrapping supports codecs whose generator uses "
            "the shared ResidualVectorQuantizer (SoundStream, Encodec, DAC, "
            f"FunCodec); got {type(quantizer).__name__} (HiFiCodec's group "
            "quantizer would need a dedicated adapter)."
        )

    comp_conf = dict(_to_plain(compression_model))
    name = comp_conf.pop("name")
    params = comp_conf.pop("params", None) or {}
    compression = build_compression_model(name, **params)
    model.codec.generator.quantizer = CompressionResidualVectorQuantizer(
        quantizer, compression, **comp_conf
    )

    freeze_codec_module(model, freeze_codec_module_name)

    if dump_config_to is not None:
        spec = {
            "codec": codec,
            "codec_conf": _to_plain(codec_conf),
            "task": task,
            "pretrained_train_config": pretrained_train_config,
            "pretrained_model_tag": pretrained_model_tag,
            "compression_model": _to_plain(compression_model),
            "freeze_codec_module_name": freeze_codec_module_name,
        }
        spec = {k: v for k, v in spec.items() if v is not None}
        dump_path = Path(dump_config_to)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)

    return model
