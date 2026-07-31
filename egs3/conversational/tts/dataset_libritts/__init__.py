"""LibriTTS data_src package for the create_dataset stage.

Same ``ConversationDataset`` as the main ``dataset`` package (the corpus
interface is the WindowRecord manifest, not a dataset class); the builder is
the LibriTTS one, so ``run.py --stages create_dataset`` prepares this corpus
when a config references ``data_src: egs3.conversational.tts.dataset_libritts``.
"""

from egs3.conversational.tts.dataset.dataset import (  # noqa: F401
    ConversationDataset as Dataset,
)
from egs3.conversational.tts.dataset.libritts_builder import (  # noqa: F401
    LibriTTSBuilder as DatasetBuilder,
)

__all__ = ["Dataset", "DatasetBuilder"]
