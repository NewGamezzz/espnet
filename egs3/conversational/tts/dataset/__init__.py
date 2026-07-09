"""SSSD conversational TTS dataset package (see PLAN-step2)."""

from .builder import SSSDBuilder as DatasetBuilder
from .dataset import ConversationDataset as Dataset
from .dataset import collate_conversations

__all__ = ["Dataset", "DatasetBuilder", "collate_conversations"]
