"""Unit tests for ConversationalTextPreprocessor (the DataOrganizer slot)."""

import pytest
import torch
from conftest import FakeTurn

from egs3.conversational.tts.dataset.preprocessing.text import (
    NEW_TOKENS,
    build_branch_texts,
    encode_tokens,
    extend_vocab,
)
from egs3.conversational.tts.dataset.preprocessor import (
    ConversationalTextPreprocessor,
    read_vocab,
)


def write_vocab(tmp_path, tokens):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return path


def sample_of(turns, num_channels):
    return {
        "window_id": "w00000",
        "num_channels": num_channels,
        "turns": list(turns),
    }


def test_literal_space_token_line(tmp_path):
    """The F5 Emilia vocab's first token is a literal space: the line must
    survive loading verbatim so id 0 stays the space character."""
    vocab = write_vocab(tmp_path, [" ", "a", "b", *NEW_TOKENS])
    assert read_vocab(vocab)[0] == " "
    pre = ConversationalTextPreprocessor(token_list=vocab)
    assert pre.token2id[" "] == 0
    sample = pre("uid", sample_of([FakeTurn(0, "spk", "a b", 0.0, 1.0)], 1))
    # <turn> a <space> b
    assert sample["text"][0].tolist() == [3, 1, 0, 2]


def test_sets_text_and_keeps_sample_keys(tmp_path, turns_3spk, base_vocab):
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    pre = ConversationalTextPreprocessor(token_list=vocab)
    sample = pre("uid", sample_of(turns_3spk, 3))
    assert sample["window_id"] == "w00000"  # untouched
    assert len(sample["text"]) == 3
    assert all(t.dtype == torch.long for t in sample["text"])


def test_matches_manual_encoding_three_channels(tmp_path, turns_3spk, base_vocab):
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    pre = ConversationalTextPreprocessor(token_list=vocab)
    sample = pre("uid", sample_of(turns_3spk, 3))
    expected = [
        encode_tokens(branch, pre.token2id)
        for branch in build_branch_texts(turns_3spk, 3)
    ]
    assert [t.tolist() for t in sample["text"]] == expected


def test_oov_fails_loudly(tmp_path):
    """After build-time normalization an OOV is a pipeline bug: no silent
    mapping to filler, unlike upstream list_str_to_idx."""
    vocab = write_vocab(tmp_path, ["a", "b", *NEW_TOKENS])
    pre = ConversationalTextPreprocessor(token_list=vocab)
    with pytest.raises(KeyError, match="not in vocab"):
        pre("uid", sample_of([FakeTurn(0, "spk", "abz", 0.0, 1.0)], 1))
