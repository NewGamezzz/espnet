"""create_dataset stage wiring: per-corpus data_src packages resolve."""

from egs3.conversational.tts.dataset.candor_builder import CandorBuilder
from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.dataset.libritts_builder import LibriTTSBuilder
from espnet3.components.data.dataset_module import load_dataset_module

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)


def test_libritts_data_src_resolves():
    module = load_dataset_module(data_src="egs3.conversational.tts.dataset_libritts")
    assert module.Dataset is ConversationDataset
    assert module.DatasetBuilder is LibriTTSBuilder


def test_candor_data_src_resolves():
    module = load_dataset_module(data_src="egs3.conversational.tts.dataset_candor")
    assert module.Dataset is ConversationDataset
    assert module.DatasetBuilder is CandorBuilder


def test_fisher_data_src_resolves():
    from egs3.conversational.tts.dataset.fisher_builder import FisherBuilder

    module = load_dataset_module(data_src="egs3.conversational.tts.dataset_fisher")
    assert module.Dataset is ConversationDataset
    assert module.DatasetBuilder is FisherBuilder
