"""Model factory for multi-compression codec fine-tuning.

Referenced from the training config as a Hydra ``_target_`` (the config
must NOT set a top-level ``task:`` key, so ``CodecSystem`` falls back to
``instantiate(config.model)``):

.. code-block:: yaml

    model:
      _target_: src.factory.build_multicomp_model
      pretrained_train_config: /path/to/baseline_exp/config.yaml
      pretrained_model_file: /path/to/baseline_exp/valid.mel_loss.ave_5best.pth
      compression_model:
        name: cosine_similarity
        params: {threshold: 1.0, mode: topk}
        min_rate: 0.2
        max_rate: 1.0
        random_rate: per_quantizer
        eval_rate: 0.5
      freeze_codec_module: none

Load order matters: pretrained weights are loaded STRICTLY into the
unwrapped model first (so the silent ``strict=False`` key-mismatch trap
cannot occur), and only then is the quantizer swapped for the
compression wrapper.  Lightning checkpoints saved during fine-tuning
therefore use the wrapped key layout.
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
    pretrained_model_file: Optional[str],
    pretrained_model_tag: Optional[str],
) -> AbsGANESPnetModel:
    """Build the unwrapped base codec model with pretrained weights loaded."""
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
        # Zoo download; key layouts of packaged models match their configs.
        from espnet2.bin.gan_codec_inference import AudioCoding

        return AudioCoding.from_pretrained(model_tag=pretrained_model_tag).model

    from espnet3.utils.task_utils import get_espnet_model

    if pretrained_train_config is not None:
        # Build from the previous run's espnet2-style config.yaml WITHOUT
        # loading weights here (build_model_from_file would, but with
        # strict=False, which silently ignores mismatched keys). Merging
        # over the task defaults also tolerates partial configs.
        with open(pretrained_train_config, encoding="utf-8") as f:
            train_args = yaml.safe_load(f)
        model = get_espnet_model(task, train_args)
    else:
        model = get_espnet_model(
            task, {"codec": codec, "codec_conf": _to_plain(codec_conf) or {}}
        )

    if pretrained_model_file is not None:
        load_model_state_strict(model, pretrained_model_file)

    return model


def load_model_state_strict(model: torch.nn.Module, model_file: str) -> None:
    """Strictly load a model-level checkpoint, rejecting Lightning layouts.

    espnet2's own loading uses ``strict=False``, so a raw Lightning
    checkpoint's ``model.``-prefixed keys would silently load nothing.
    """
    state_dict = torch.load(model_file, map_location="cpu")
    if any(k.startswith("model.") for k in state_dict):
        raise ValueError(
            f"'{model_file}' contains 'model.'-prefixed keys (a raw Lightning "
            "checkpoint). Use the averaged checkpoint (e.g. "
            "valid.mel_loss.ave_5best.pth), whose prefix is already stripped."
        )
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
    """Build a pretrained codec wrapped with the multi-compression quantizer.

    Args:
        compression_model: dict with ``name`` (registry key), optional
            ``params`` for the compression model itself, and the wrapper's
            rate settings (``min_rate``, ``max_rate``, ``random_rate``,
            ``eval_rate``).
        codec / codec_conf: fresh GANCodecTask model spec (mutually
            exclusive with the pretrained_* sources).
        task: espnet2 task path used with ``codec``/``codec_conf``.
        pretrained_train_config: espnet2-style config.yaml of a previous
            run (built via ``save_espnet_config``); combine with
            ``pretrained_model_file`` to load its weights.
        pretrained_model_file: averaged checkpoint (model-level keys,
            e.g. ``valid.mel_loss.ave_5best.pth``), loaded strictly into
            the unwrapped model.
        pretrained_model_tag: espnet_model_zoo tag (e.g.
            ``espnet/libritts_encodec_24k``); weights come with the tag.
        freeze_codec_module_name: ``none`` | ``encoder`` | ``decoder``.
        dump_config_to: when set (training config points it at
            ``${exp_dir}/multicomp_model.yaml``), the resolved model spec
            minus ``pretrained_model_file`` is written there so inference
            can rebuild the wrapped architecture and load the fine-tuned
            checkpoint on top.  This substitutes for ``save_espnet_config``,
            which ``CodecSystem.train`` skips on the task-less Hydra path.

    Returns:
        ``ESPnetGANCodecModel`` (an ``AbsGANESPnetModel``, so
        ``CodecSystem`` selects the GAN trainer automatically).
    """
    model = _build_base_model(
        codec=codec,
        codec_conf=codec_conf,
        task=task,
        pretrained_train_config=pretrained_train_config,
        pretrained_model_file=pretrained_model_file,
        pretrained_model_tag=pretrained_model_tag,
    )

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
            # pretrained_model_file intentionally omitted: at inference the
            # fine-tuned checkpoint supplies all weights.
            "compression_model": _to_plain(compression_model),
            "freeze_codec_module_name": freeze_codec_module_name,
        }
        spec = {k: v for k, v in spec.items() if v is not None}
        dump_path = Path(dump_config_to)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)

    return model
