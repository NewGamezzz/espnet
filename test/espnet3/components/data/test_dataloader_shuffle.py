"""The config's shuffle flag must reach SequenceIterFactory.build_iter."""

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from espnet3.components.data.dataloader import DataLoaderBuilder


def test_build_iter_receives_config_shuffle(monkeypatch):
    """_build_iter_factory must not force shuffle=False."""
    recorded = {}

    class FakeFactory:
        def build_iter(self, epoch, shuffle=None):
            recorded["epoch"] = epoch
            recorded["shuffle"] = shuffle
            return "ITERATOR"

    monkeypatch.setattr(
        "espnet3.components.data.dataloader.build_batch_sampler",
        lambda **kw: [(0, 1), (2, 3)],
    )

    def fake_instantiate(cfg, *args, **kwargs):
        # The batch list must never travel through instantiate: Hydra merges
        # its arguments into an OmegaConf config, which takes minutes for the
        # batch list of a large corpus. It is handed to the partial instead.
        recorded["instantiate_args"] = args
        recorded["instantiate_kwargs"] = kwargs

        def factory(dataset, batches):
            recorded["batches"] = batches
            return FakeFactory()

        return factory

    monkeypatch.setattr(
        "espnet3.components.data.dataloader.instantiate", fake_instantiate
    )

    builder = DataLoaderBuilder(
        dataset=MagicMock(),
        config=OmegaConf.create({}),
        collate_fn=None,
        num_device=1,
        epoch=3,
    )

    factory_config = {
        "_target_": "espnet2.iterators.sequence_iter_factory." "SequenceIterFactory",
        "batches": {},
        "shuffle": True,
    }
    assert builder._build_iter_factory(factory_config) == "ITERATOR"

    # shuffle=None lets SequenceIterFactory fall back to self.shuffle,
    # which is the config value. A hard False would ignore the config.
    assert (
        recorded["shuffle"] is None
    ), "dataloader must not override the config's shuffle setting"
    assert recorded["epoch"] == 3
    assert recorded["instantiate_args"] == ()
    assert recorded["instantiate_kwargs"] == {"_partial_": True}
    assert recorded["batches"] == [(0, 1), (2, 3)]
