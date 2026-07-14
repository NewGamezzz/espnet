"""ConversationalLightningModule logging: per-channel keys stay unsynced."""

import pytest
import torch
from omegaconf import OmegaConf

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)

from egs3.conversational.tts.src.lit_module import ConversationalLightningModule


class _RecordingLogDict:
    def __init__(self):
        self.calls = []

    def __call__(self, metrics, **kwargs):
        self.calls.append((metrics, kwargs))


def _bare_module():
    """Skip the full __init__ (it instantiates DataOrganizer); _log_stats
    needs only ``_trainer`` and ``log_dict``."""
    module = ConversationalLightningModule.__new__(ConversationalLightningModule)
    recorder = _RecordingLogDict()
    module.__dict__["_trainer"] = object()
    module.__dict__["log_dict"] = recorder
    return module, recorder


@pytest.mark.parametrize("mode", ["train", "valid"])
def test_log_stats_routes_per_channel_keys_unsynced(mode):
    """loss_ch* keys vary with the batch's max N, so they must never enter
    the sync_dist path (ragged key sets across DDP ranks deadlock NCCL)."""
    module, recorder = _bare_module()
    stats = {
        "loss": torch.tensor(1.0),
        "loss_ch0": torch.tensor(0.5),
        "loss_ch1": torch.tensor(0.7),
    }
    module._log_stats(mode, stats, weight=torch.tensor(2.0))

    assert len(recorder.calls) == 2
    (synced_metrics, synced_kwargs), (raw_metrics, raw_kwargs) = recorder.calls
    assert set(synced_metrics) == {f"{mode}/loss"}
    assert synced_kwargs["sync_dist"] is (mode == "valid")
    assert set(raw_metrics) == {f"{mode}/loss_ch0", f"{mode}/loss_ch1"}
    assert raw_kwargs["sync_dist"] is False
    # The caller's stats dict is not mutated.
    assert set(stats) == {"loss", "loss_ch0", "loss_ch1"}


def test_log_stats_without_per_channel_keys():
    module, recorder = _bare_module()
    module._log_stats("valid", {"loss": torch.tensor(1.0)}, weight=None)
    assert len(recorder.calls) == 1  # no second (unsynced) call


class _StubOrganizer:
    """Minimal DataOrganizer stand-in for constructing the module."""

    def __init__(self):
        self.train = ["window"]
        self.valid = ["window"]

    def log_summary(self, logger):
        pass


def test_init_marks_sampler_as_self_sharding():
    """ConversationBatchSampler strides batches by rank itself, so the module
    must set is_espnet_sampler so the espnet3 trainer hands Lightning
    use_distributed_sampler=False (otherwise DDP crashes trying to inject a
    DistributedSampler into the non-BatchSampler batch sampler)."""
    config = OmegaConf.create(
        {
            "dataset": {
                "_target_": f"{_StubOrganizer.__module__}._StubOrganizer",
            },
            "dataloader": {"train": {}, "valid": {}},
        }
    )
    module = ConversationalLightningModule(torch.nn.Linear(1, 1), config)
    assert module.is_espnet_sampler is True
