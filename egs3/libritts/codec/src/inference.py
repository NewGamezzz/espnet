"""Inference helpers for the LibriTTS codec recipe."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def build_output(data, model_output, idx):
    """Format one AudioCoding roundtrip result for SCP/artifact writing.

    Args:
        data: Dataset item (``inference: true`` mode), providing ``utt_id``
            and the ground-truth ``wav_path``.
        model_output: Dict returned by
            ``espnet2.bin.gan_codec_inference.AudioCoding.__call__``,
            containing the resynthesized waveform under ``resyn_audio``.
        idx: Dataset index, used as a fallback identifier.

    Returns:
        Dict with ``utt_id``, the ground-truth wav path under ``ref``, and
        the resynthesized waveform under ``wav`` (1-D float32 numpy array,
        materialized as a WAV file via ``output_artifacts``).
    """
    utt_id = str(data.get("utt_id", idx))
    ref = str(data.get("wav_path", ""))  # ground truth wav path
    wav = model_output.get("resyn_audio")
    if wav is None:
        raise RuntimeError("Codec inference output does not contain 'resyn_audio'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "ref": ref, "wav": wav}


class MultiCompressionAudioCoding:
    """Encode -> decode roundtrip for a multi-compression codec.

    Analog of ``espnet2.bin.gan_codec_inference.AudioCoding`` for models
    built by ``src.factory.build_multicomp_model``.  The wrapped
    architecture comes from exactly one of two sources:

    - ``train_config``: the model spec the factory dumped during
      fine-tuning (``dump_config_to`` -> ``multicomp_model.yaml``).
    - ``model_conf``: inline factory kwargs (``compression_model`` plus
      a codec source such as ``pretrained_train_config``), for
      evaluating a codec that was never fine-tuned with the wrapper
      (e.g. rate-sweeping baseline weights).

    The weights ALWAYS come from ``model_file`` and are loaded strictly:
    fine-tuned checkpoints match the wrapped key layout directly, and
    baseline (unwrapped-layout) checkpoints are remapped by the
    wrapper's load hook; Lightning checkpoints are unwrapped
    automatically.  ``model_conf`` must therefore not name weight
    sources (``pretrained_model_file``/``pretrained_model_tag``).

    The compression ``rate`` and the optional ``anchor_start_layer``
    (layers at/after it constrain their boundaries to the union of the
    earlier layers' boundaries) are delivered to the quantizer wrapper via
    its ``set_inference_*`` state around the unmodified ``codec.encode``
    call (which forwards no extra kwargs), and can be overridden per call
    via ``decode_conf={"rate": ..., "anchor_start_layer": ...}``.
    """

    def __init__(
        self,
        train_config: Optional[str] = None,
        model_file: Optional[str] = None,
        model_conf: Optional[Dict[str, Any]] = None,
        rate: Optional[float] = None,
        anchor_start_layer: Optional[int] = None,
        target_bandwidth: Optional[float] = None,
        dtype: str = "float32",
        device: Union[str, torch.device] = "cpu",
    ):
        from .factory import _to_plain, build_multicomp_model, load_model_state_strict

        if model_file is None:
            raise ValueError(
                "'model_file' is required: it names the evaluated weights "
                "(fine-tuned or baseline checkpoint)."
            )
        if (train_config is None) == (model_conf is None):
            raise ValueError(
                "Specify exactly one architecture source: 'train_config' "
                "(the multicomp_model.yaml spec dumped during fine-tuning) "
                "or 'model_conf' (inline factory kwargs, e.g. for "
                "evaluating baseline weights without fine-tuning)."
            )

        if train_config is not None:
            with open(train_config, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
        else:
            spec = dict(_to_plain(model_conf))
            forbidden = {"pretrained_model_file", "pretrained_model_tag"} & set(spec)
            if forbidden:
                raise ValueError(
                    f"model_conf must not contain weight sources "
                    f"({sorted(forbidden)}): weights always come from "
                    "'model_file'."
                )

        model = build_multicomp_model(**spec)
        load_model_state_strict(model, model_file)
        model.to(device).to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.quantizer = model.codec.generator.quantizer
        self.device = device
        self.dtype = dtype
        self.rate = rate
        self.anchor_start_layer = anchor_start_layer
        self.target_bandwidth = target_bandwidth

    @torch.no_grad()
    def __call__(
        self,
        audio: Union[torch.Tensor, np.ndarray, None] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
        encode_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the codec roundtrip; mirrors ``AudioCoding.__call__``."""
        assert audio is not None, "Audio is invalid, input a valid audio."

        cfg: Dict[str, Any] = {
            "rate": self.rate,
            "anchor_start_layer": self.anchor_start_layer,
            "target_bw": self.target_bandwidth,
        }
        if decode_conf is not None:
            cfg.update(decode_conf)

        audio = torch.as_tensor(audio, dtype=getattr(torch, self.dtype)).to(self.device)

        self.quantizer.set_inference_rate(cfg["rate"])
        self.quantizer.set_inference_anchor_start_layer(cfg["anchor_start_layer"])
        try:
            codes = self.model.encode(audio, target_bw=cfg["target_bw"])
        finally:
            self.quantizer.reset_inference_rate()
            self.quantizer.reset_inference_anchor_start_layer()

        output_dict = dict(codes=codes)
        if not encode_only:
            if audio.dim() == 1:
                resyn_audio = self.model.decode(codes).view(-1)
            else:
                resyn_audio = self.model.decode(codes)
            output_dict.update(resyn_audio=resyn_audio)
        return output_dict
