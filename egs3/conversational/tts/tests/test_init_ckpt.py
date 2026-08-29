"""Weights-only init of the assembled model from a Lightning checkpoint."""

import pytest
import torch

from egs3.conversational.tts.src.build_model import load_init_checkpoint

from .conftest import ext_vocab_file  # noqa: F401  (fixture)
from .test_build_model import build_tiny


def _ckpt_from(model, tmp_path, *, with_ema=True, perturb_ema=True):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = {"state_dict": sd, "global_step": 123, "epoch": 4}
    if with_ema:
        ema = {
            f"ema_model.{k}": (v + 1.0 if perturb_ema and v.is_floating_point() else v)
            for k, v in sd.items()
        }
        ema["initted"] = torch.tensor(True)
        ema["step"] = torch.tensor(123)
        ckpt["ema_model_state_dict"] = ema
    path = tmp_path / "step123.ckpt"
    torch.save(ckpt, path)
    return path


def test_init_from_ema_loads_ema_tensors(ext_vocab_file, tmp_path):  # noqa: F811
    src = build_tiny(ext_vocab_file)
    path = _ckpt_from(src, tmp_path)
    dst = build_tiny(ext_vocab_file)
    load_init_checkpoint(dst, path, from_ema=True)
    for k, v in src.state_dict().items():
        expected = v + 1.0 if v.is_floating_point() else v
        torch.testing.assert_close(dst.state_dict()[k], expected)


def test_init_from_raw_loads_state_dict(ext_vocab_file, tmp_path):  # noqa: F811
    src = build_tiny(ext_vocab_file)
    path = _ckpt_from(src, tmp_path)
    dst = build_tiny(ext_vocab_file)
    load_init_checkpoint(dst, path, from_ema=False)
    for k, v in src.state_dict().items():
        torch.testing.assert_close(dst.state_dict()[k], v)


def test_init_from_ema_requires_ema_block(ext_vocab_file, tmp_path):  # noqa: F811
    src = build_tiny(ext_vocab_file)
    path = _ckpt_from(src, tmp_path, with_ema=False)
    with pytest.raises(KeyError, match="ema_model_state_dict"):
        load_init_checkpoint(build_tiny(ext_vocab_file), path, from_ema=True)


def test_init_strict_mismatch_raises(ext_vocab_file, tmp_path):  # noqa: F811
    src = build_tiny(ext_vocab_file)
    path = _ckpt_from(src, tmp_path)
    ckpt = torch.load(path, weights_only=False)
    del ckpt["state_dict"][next(iter(ckpt["state_dict"]))]
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError):
        load_init_checkpoint(build_tiny(ext_vocab_file), path, from_ema=False)


def test_builder_kwargs_wire_init(ext_vocab_file, tmp_path):  # noqa: F811
    src = build_tiny(ext_vocab_file)
    path = _ckpt_from(src, tmp_path, perturb_ema=False)
    dst = build_tiny(ext_vocab_file, init_ckpt=str(path), init_from_ema=True)
    for k, v in src.state_dict().items():
        torch.testing.assert_close(dst.state_dict()[k], v)
