"""MmapCheckpointIO: resume restores must not materialize the ckpt on the heap.

Background (2026-07-28 Delta investigation): plain torch.load() during
fit(ckpt_path=...) left ~5 GB of unreturned heap per rank (9.5 GB RSS vs
4.6 GB fresh), pushing nodes past their page-reclaim threshold and
livelocking the forward pass on Lustre page faults. Loading with
mmap=True keeps the checkpoint file-backed (login-node measurement:
0.42 GB peak RSS vs 6.58 GB plain on the real 6.9 GB checkpoint).
"""

import inspect

import pytest
import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO

from .conftest import REPO_ROOT

from egs3.conversational.tts.src.checkpoint_io import MmapCheckpointIO


def test_signature_extends_torch_checkpoint_io():
    """Lightning's checkpoint connector may call load_checkpoint with any
    argument combination the parent accepts - in 2.6.5 that includes
    ``weights_only`` (an override without it TypeErrors at resume). The
    override's parameters must therefore start with the parent's, in
    order, and must include ``weights_only`` explicitly."""
    parent = list(inspect.signature(TorchCheckpointIO.load_checkpoint).parameters)
    ours = list(inspect.signature(MmapCheckpointIO.load_checkpoint).parameters)
    assert ours[: len(parent)] == parent
    assert "weights_only" in ours


def test_local_file_loads_with_mmap_and_trusted_weights_only(monkeypatch, tmp_path):
    """The whole point of the class: local files load with mmap=True. A
    ``weights_only=None`` from the connector must become False (Lightning
    checkpoints carry loops/callbacks/hyperparameters, and torch's own
    default for None is True since 2.6)."""
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {"w": torch.ones(2)}}, ckpt)

    seen = {}
    real_load = torch.load

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    MmapCheckpointIO().load_checkpoint(str(ckpt))

    assert seen["mmap"] is True
    assert seen["weights_only"] is False


def test_explicit_weights_only_is_forwarded(monkeypatch, tmp_path):
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"w": torch.ones(2)}, ckpt)

    seen = {}
    real_load = torch.load

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    MmapCheckpointIO().load_checkpoint(str(ckpt), weights_only=True)

    assert seen["weights_only"] is True


def test_round_trip_preserves_checkpoint_contents(tmp_path):
    """mmap-loaded tensors must compare equal to what was saved - the
    restore path copies them into live storages via load_state_dict."""
    payload = {
        "state_dict": {"w": torch.arange(4.0)},
        "epoch": 3,
        "loops": {"fit_loop": {"state": 1}},
    }
    ckpt = tmp_path / "full.ckpt"
    torch.save(payload, ckpt)

    loaded = MmapCheckpointIO().load_checkpoint(str(ckpt))

    assert loaded["epoch"] == 3
    assert loaded["loops"] == payload["loops"]
    assert torch.equal(loaded["state_dict"]["w"], payload["state_dict"]["w"])


def test_missing_file_still_raises_file_not_found(tmp_path):
    """Non-files fall through to the parent, which keeps Lightning's
    FileNotFoundError contract for bad ckpt_path values."""
    with pytest.raises(FileNotFoundError):
        MmapCheckpointIO().load_checkpoint(str(tmp_path / "absent.ckpt"))


def test_training_config_installs_mmap_checkpoint_io():
    """The fix only takes effect if the recipe config actually mounts the
    plugin; espnet3 instantiates trainer.plugins entries by _target_."""
    import yaml

    config_path = (
        REPO_ROOT / "egs3" / "conversational" / "tts" / "conf" / "training_poc.yaml"
    )
    trainer = yaml.safe_load(config_path.read_text())["trainer"]
    targets = [entry["_target_"] for entry in trainer["plugins"]]
    assert "egs3.conversational.tts.src.checkpoint_io.MmapCheckpointIO" in targets


def _bare_module():
    from egs3.conversational.tts.src.lit_module import ConversationalLightningModule

    return ConversationalLightningModule.__new__(ConversationalLightningModule)


def test_on_train_start_trims_heap(monkeypatch):
    """After restore, on_train_start hands fragmented heap pages back to
    the OS via glibc malloc_trim(0)."""
    calls = []

    class _Libc:
        def malloc_trim(self, arg):
            calls.append(arg)

    monkeypatch.setattr("ctypes.CDLL", lambda name: _Libc())
    _bare_module().on_train_start()
    assert calls == [0]


def test_on_train_start_survives_non_glibc(monkeypatch):
    """macOS dev boxes have no libc.so.6; the hook must swallow the
    OSError rather than kill training."""

    def _raise(name):
        raise OSError("no libc.so.6")

    monkeypatch.setattr("ctypes.CDLL", _raise)
    _bare_module().on_train_start()  # must not raise
