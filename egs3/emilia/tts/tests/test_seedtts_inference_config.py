"""Pin the Seed-TTS inference config to the schema DataOrganizer actually
resolves.

Regression for a schema bug caught in review: a flat
``dataset: {test_en: [...], test_zh: [...], test_hard: [...]}`` shape with
bare ``_target_: ...SeedTTSDataset`` entries does not match
``DataOrganizer``, which requires ``dataset.test`` to be a *list* of
``{name, data_src, data_src_args}`` entries (see
``espnet3/components/data/data_organizer.py`` and
``espnet3/systems/base/inference.py``, which iterates
``config.dataset.test``). Worse, ``instantiate_dataset_reference`` resolves
the dataset class via ``getattr(module, "Dataset")`` unconditionally
(``espnet3/components/data/dataset_module.py``), so a bare ``_target_``
inside ``data_src_args`` is never consulted -- the module's ``Dataset``
alias is what actually gets instantiated.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egs3.emilia.tts.dataset.seedtts import SeedTTSDataset

CONF = Path(__file__).resolve().parents[1] / "conf" / "inference_f5_seedtts.yaml"


@pytest.fixture
def cfg():
    return OmegaConf.load(CONF)


def test_dataset_uses_data_organizer(cfg):
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )


def test_test_sets_are_a_list_of_three_named_entries(cfg):
    names = [entry.name for entry in cfg.dataset.test]
    assert names == ["test_en", "test_zh", "test_hard"]


def test_every_entry_selects_the_seedtts_dataset_module(cfg):
    for entry in cfg.dataset.test:
        assert entry.data_src == "egs3.emilia.tts.dataset.seedtts"


def test_seedtts_module_exposes_dataset_alias():
    """The mechanism DataOrganizer actually relies on: getattr(module, "Dataset")."""
    import egs3.emilia.tts.dataset.seedtts as seedtts_module

    assert seedtts_module.Dataset is SeedTTSDataset


def test_device_and_train_config_are_set(cfg):
    assert cfg.device == "cuda"
    # Raw (unresolved) interpolation string: must reference the Base config
    # from Task 7, not the LibriTTS small config.
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw["model"]["train_config"] == (
        "${recipe_dir}/conf/training_f5_tts_base.yaml"
    )
