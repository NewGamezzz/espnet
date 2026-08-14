"""ConversationalLightningModule logging: per-channel keys stay unsynced."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)

from egs3.conversational.tts.src.lit_module import (
    ConversationalLightningModule,
    PackedConversationCollator,
    PlannedWindowView,
)
from egs3.conversational.tts.src.sampler import ConversationBatchSampler


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


def test_training_config_has_no_per_epoch_reload_and_keeps_sanity_probe():
    """reload=1 made resume build two train loaders (sampler logs epoch=5
    then epoch=6, Delta job 20532548); reshuffling now rides
    ConversationBatchSampler.set_epoch. The sanity probe (num_sanity_val_steps)
    is kept ON to fail fast on broken val/train paths - its dataloader
    workers are shut down by ConversationalLightningModule.on_validation_end
    (gated on trainer.sanity_checking) instead of being disabled.

    Do NOT set reload=1 to make Lightning re-query the online sampler's
    per-epoch batch count (Task 10 plan-review item): that would reintroduce
    the resume bug above. src/sampler.py's docstring documents (with numbers
    from test_training_smoke.py's
    test_online_sampler_survives_two_epochs_with_varying_batch_counts) that
    reload=0 keeps the fit safe - no hang, no truncated fit - even though a
    later epoch's real online-planned batch count can be undercounted
    relative to epoch 0's frozen total."""
    import yaml
    from .conftest import REPO_ROOT

    config_path = (
        REPO_ROOT / "egs3" / "conversational" / "tts" / "conf" / "training_poc.yaml"
    )
    trainer = yaml.safe_load(config_path.read_text())["trainer"]
    assert trainer["reload_dataloaders_every_n_epochs"] == 0
    assert trainer["num_sanity_val_steps"] == 2


def _bare_module_with_trainer(sanity_checking: bool):
    """Same __new__ bypass as ``_bare_module`` (skips DataOrganizer setup),
    but stubs enough of the ``trainer`` property's internals
    (``_fabric``, ``_jit_is_scripting``, ``_trainer``) for
    ``self.trainer.sanity_checking`` to resolve, since ``on_validation_end``
    reads it through the real property rather than ``_trainer`` directly."""
    module = ConversationalLightningModule.__new__(ConversationalLightningModule)
    module.__dict__["_fabric"] = None
    module.__dict__["_jit_is_scripting"] = False
    module.__dict__["_trainer"] = SimpleNamespace(sanity_checking=sanity_checking)
    return module


@pytest.mark.parametrize("sanity_checking", [True, False])
def test_on_validation_end_releases_iterator_only_during_sanity(sanity_checking):
    """Exercises the real production on_validation_end wrapper (super() call
    + trainer.sanity_checking gate) end to end, not just the private
    _release_sanity_val_iterator body the integration test in
    test_sampler.py delegates to. Records calls via an instance-level stub
    (a plain function shadows the class method, no monkeypatch needed)."""
    module = _bare_module_with_trainer(sanity_checking)
    calls = []
    module._release_sanity_val_iterator = lambda: calls.append(True)

    module.on_validation_end()

    assert len(calls) == (1 if sanity_checking else 0)


# --------------------------------------------------------------------------
# Task 10: PlannedWindowView routes (component_idx, WindowRecord) specs from
# ConversationBatchSampler's online-mode __iter__ to the right
# ConversationDataset, mirroring CombinedDataset.__getitem__'s transform/
# preprocessor/collator application (which only understands int/str indices,
# not specs - see PlannedWindowView's docstring in src/lit_module.py).
# --------------------------------------------------------------------------


class TestPlannedWindowView:
    def test_spec_routing_matches_int_routing(self, combined_two_corpora):
        view = PlannedWindowView(combined_two_corpora)
        comp = combined_two_corpora.datasets[1]
        record = comp.records[0]
        via_spec = view[(1, record)]
        # int index of the same record through CombinedDataset
        offset = len(combined_two_corpora.datasets[0])
        via_int = combined_two_corpora[offset + 0]
        assert via_spec.keys() == via_int.keys()
        assert via_spec["window_id"] == via_int["window_id"]
        assert torch.equal(via_spec["text"][0], via_int["text"][0])  # preprocessor ran

    def test_bare_dataset_supported(self, bare_conversation_dataset):
        view = PlannedWindowView(bare_conversation_dataset)
        rec = bare_conversation_dataset.records[0]
        assert view[(0, rec)]["window_id"] == rec.window_id

    def test_len_delegates_to_underlying_dataset(self, combined_two_corpora):
        view = PlannedWindowView(combined_two_corpora)
        assert len(view) == len(combined_two_corpora)

    def test_collator_uid_contract_matches_combined_dataset(self, combined_two_corpora):
        """When use_espnet_collator is set (mirroring what _packed_dataloader
        does on the underlying dataset before wrapping it), the view returns
        (window_id, sample) - the same (uid, sample) shape CombinedDataset's
        int path returns under the same flag."""
        combined_two_corpora.use_espnet_collator = True
        view = PlannedWindowView(combined_two_corpora)
        comp = combined_two_corpora.datasets[0]
        record = comp.records[0]
        uid, sample = view[(0, record)]
        assert uid == record.window_id
        assert "text" in sample
        combined_two_corpora.use_espnet_collator = False


# --------------------------------------------------------------------------
# Task 10: _packed_dataloader wraps the dataset in PlannedWindowView and
# constructs ConversationBatchSampler with online=(mode == "train") - valid
# stays on the frozen per-record plan, train replans fresh every epoch.
# --------------------------------------------------------------------------


def _bare_module_for_loader(dataset) -> ConversationalLightningModule:
    """Same __new__ bypass as _bare_module (skips DataOrganizer setup):
    _packed_dataloader only reads self.config/self.collate_fn/_initial_epoch,
    all stubbed directly instead of going through the full hydra config path
    a real DataOrganizer would need."""
    module = ConversationalLightningModule.__new__(ConversationalLightningModule)
    module.__dict__["_trainer"] = None
    module.__dict__["collate_fn"] = PackedConversationCollator()
    loader_kwargs = dict(
        batch_bins=10**9,  # everything fits in one batch; only wiring matters
        min_batch_size=1,
        num_workers=0,
    )
    module.__dict__["config"] = OmegaConf.create(
        {
            "seed": 0,
            "dataloader": {
                "train": {**loader_kwargs, "shuffle": True},
                "valid": {**loader_kwargs, "shuffle": False},
            },
        }
    )
    return module


class TestPackedDataloaderWiring:
    def test_train_dataloader_wraps_view_and_is_online(self, bare_conversation_dataset):
        module = _bare_module_for_loader(bare_conversation_dataset)
        loader = module._packed_dataloader(bare_conversation_dataset, "train")
        assert isinstance(loader.dataset, PlannedWindowView)
        assert isinstance(loader.batch_sampler, ConversationBatchSampler)
        assert loader.batch_sampler.online is True

    def test_valid_dataloader_wraps_view_and_is_frozen(self, bare_conversation_dataset):
        module = _bare_module_for_loader(bare_conversation_dataset)
        loader = module._packed_dataloader(bare_conversation_dataset, "valid")
        assert isinstance(loader.dataset, PlannedWindowView)
        assert isinstance(loader.batch_sampler, ConversationBatchSampler)
        assert loader.batch_sampler.online is False

    def test_dataloader_yields_a_real_batch(self, combined_two_corpora):
        """End-to-end: the DataLoader built by _packed_dataloader actually
        produces a collated batch through PlannedWindowView + load_window +
        the preprocessor + PackedConversationCollator, not just the right
        wiring types (combined_two_corpora carries a real preprocessor, so
        the "text" key collate_conversations needs is actually present)."""
        module = _bare_module_for_loader(combined_two_corpora)
        loader = module._packed_dataloader(combined_two_corpora, "valid")
        window_ids, batch = next(iter(loader))
        expected_ids = {
            r.window_id for c in combined_two_corpora.datasets for r in c.records
        }
        assert set(window_ids) <= expected_ids
        assert batch["speech"].shape[0] >= 1
