"""Runtime text tokenization for ``ConversationDataset`` items.

House style keeps datasets vocab-agnostic: tokenization lives in the
``DataOrganizer`` ``preprocessor:`` slot, configured with a ``token_list``
path (see ``egs3/libritts/tts/conf/training_f5_tts_small.yaml``).
``CommonPreprocessor`` cannot be reused because this recipe's "text" is N
parallel per-channel streams derived from turn structure, not one flat
string, so this small recipe-local preprocessor applies the masking scheme
of ``preprocessing/text.py`` and encodes ids against the extended vocab.

Tokenization stays exactly the pretrained F5TTS_Base convention: raw
characters, case preserved, id = vocab line index, and no cleaner or
``<blank>/<unk>/<sos/eos>`` symbols (those belong to from-scratch recipes;
adding them here would corrupt id alignment with the pretrained
text-embedding matrix).  Batches pad text ids with -1: the model shifts ids
by +1 and uses 0 as its internal filler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from espnet2.train.preprocessor import AbsPreprocessor

from .preprocessing.text import (
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    build_branch_texts,
    encode_tokens,
    make_token2id,
)


def read_vocab(path: str | Path) -> list[str]:
    # Verbatim lines: the line index IS the token id (a token may be a literal
    # space, as in the F5 Emilia vocab, so no whitespace filtering).
    return Path(path).read_text(encoding="utf-8").splitlines()


class ConversationalTextPreprocessor(AbsPreprocessor):
    """Encode a conversation item's turns into per-branch token-id tensors.

    ``DataOrganizer`` calls this as ``preprocessor(uid, sample)``.  The
    sample's ``turns`` come from ``ConversationDataset`` with ``channel``
    already remapped to the post-permutation row index, so branch ``i`` of
    the output aligns with ``speech`` row ``i`` for any permutation.  Sets
    ``sample["text"]`` to a list of N variable-length int64 tensors, the
    contract ``collate_conversations`` packs.

    Chunk-task samples (carrying ``prompt_frames``/``prev_frames`` int
    keys) get the same
    ``[SPEAKER_PROMPT_TOKEN] * prompt_frames + [PREV_CHUNK_TOKEN] *
    prev_frames`` prefix on every branch, ahead of its normal turn-marked
    text, so the text stream has one marker per mel frame over the
    ``speech`` tensor's P/H span.  Ordinary infill samples carry neither
    key and get no prefix.
    """

    def __init__(self, token_list: str | Path, train: bool = False) -> None:
        super().__init__(train)
        self.token_list = Path(token_list)
        self.token2id = make_token2id(read_vocab(self.token_list))

    def __call__(self, uid: str, data: dict[str, Any]) -> dict[str, Any]:
        branch_tokens = build_branch_texts(data["turns"], data["num_channels"])
        if "prompt_frames" in data:
            # Chunk-task sample: P/H are audio-only, so they get a flat run
            # of one marker per mel frame - identical on every branch, since
            # the conditioning audio isn't attributed to a speaker until the
            # target region begins.  prev_frames is always set alongside
            # prompt_frames, so a direct lookup fails loudly on an
            # inconsistent caller.
            prefix = [SPEAKER_PROMPT_TOKEN] * data["prompt_frames"] + [
                PREV_CHUNK_TOKEN
            ] * data["prev_frames"]
            branch_tokens = [prefix + tokens for tokens in branch_tokens]
        # encode_tokens fails loudly on OOV: after build-time normalization
        # an unknown token is a pipeline bug, not user input to be cleaned.
        data["text"] = [
            torch.tensor(encode_tokens(tokens, self.token2id), dtype=torch.long)
            for tokens in branch_tokens
        ]
        return data
