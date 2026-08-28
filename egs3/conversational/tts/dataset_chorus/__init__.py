"""NSF Chorus data_src package for the create_dataset stage.

Same ``ConversationDataset`` as the main ``dataset`` package (the corpus
interface is the session manifest, not a dataset class); the builder is the
Chorus one, so ``run.py --stages create_dataset`` prepares this corpus when a
config references ``data_src: egs3.conversational.tts.dataset_chorus``.
"""

from egs3.conversational.tts.dataset.chorus_builder import (  # noqa: F401
    ChorusBuilder as DatasetBuilder,
)
from egs3.conversational.tts.dataset.dataset import (  # noqa: F401
    ConversationDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
