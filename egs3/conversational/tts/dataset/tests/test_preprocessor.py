"""Unit tests for ConversationalTextPreprocessor (the DataOrganizer slot)."""

import pytest
import torch
from .conftest import FakeTurn

from egs3.conversational.tts.dataset.preprocessing.text import (
    NEW_TOKENS,
    SPEAKER_PROMPT_TOKEN,
    TURN_FILL_TOKEN,
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


@pytest.fixture
def preproc(tmp_path, base_vocab):
    """Preprocessor built against the shared small extended vocab (base_vocab
    + NEW_TOKENS, which includes <turn_fill>)."""
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    return ConversationalTextPreprocessor(token_list=vocab)


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


def test_chunk_task_prefix_per_frame(tmp_path, turns_3spk, base_vocab):
    """Chunk-task samples (Task 6's ``prompt_frames``/``prev_frames`` keys)
    get an identical [SPEAKER_PROMPT_TOKEN]*prompt_frames +
    [PREV_CHUNK_TOKEN]*prev_frames prefix on every branch, ahead of that
    branch's normal build_branch_texts output."""
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    pre = ConversationalTextPreprocessor(token_list=vocab)
    sample = sample_of(turns_3spk, 3)
    sample["prompt_frames"] = 5
    sample["prev_frames"] = 3
    out = pre("uid", sample)
    sp = pre.token2id["<speaker_prompt>"]
    pc = pre.token2id["<prev_chunk>"]
    cond = 5 + 3
    for t in out["text"]:
        assert t[:5].tolist() == [sp] * 5
        assert t[5:8].tolist() == [pc] * 3
    a, b = out["text"][0], out["text"][1]
    assert not torch.equal(a[8:], b[8:])  # target region still branch-specific
    expected = [
        encode_tokens(branch, pre.token2id)
        for branch in build_branch_texts(turns_3spk, 3)
    ]
    assert [t[cond:].tolist() for t in out["text"]] == expected


def test_chunk_task_prefix_prompt_only(tmp_path, turns_3spk, base_vocab):
    """``prev_frames == 0`` (prompt_only plan): prefix is prompt tokens only,
    immediately followed by the ordinary <turn>-led branch text."""
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    pre = ConversationalTextPreprocessor(token_list=vocab)
    sample = sample_of(turns_3spk, 3)
    sample["prompt_frames"] = 4
    sample["prev_frames"] = 0
    out = pre("uid", sample)
    sp = pre.token2id["<speaker_prompt>"]
    turn = pre.token2id["<turn>"]
    for t in out["text"]:
        assert t[:4].tolist() == [sp] * 4
        assert t[4].item() == turn


def test_infill_has_no_prefix(tmp_path, turns_3spk, base_vocab):
    """Samples without the chunk-task keys (ordinary infill) are byte-
    identical to the pre-Task-7 output: no prefix, first token is <turn>."""
    vocab = write_vocab(tmp_path, extend_vocab(base_vocab))
    pre = ConversationalTextPreprocessor(token_list=vocab)
    out = pre("uid", sample_of(turns_3spk, 3))
    assert out["text"][0][0].item() == pre.token2id["<turn>"]


class TestTimestampPreprocessor:
    """Mode T (Task 6): timestamp-aligned target text, prefix composition
    unchanged. Turns are the ``FakeTurn`` fixture used throughout this
    suite (duck-typed stand-in for sssd.Turn)."""

    def _data(self, turns, extra=None):
        d = {"turns": turns, "num_channels": 2}
        d.update(extra or {})
        return d

    def test_mode_t_text_length_equals_target_frames(self, preproc):
        turns = [FakeTurn(channel=0, speaker="a", text="hi", start=10.5, end=11.5)]
        data = self._data(
            turns, {"timestamp_text": True, "target_t0": 10.0, "target_frames": 187}
        )
        out = preproc("uid", data)
        assert all(t.shape[0] == 187 for t in out["text"])

    def test_mode_t_composes_with_chunk_prefix(self, preproc):
        turns = [FakeTurn(channel=0, speaker="a", text="hi", start=10.5, end=11.5)]
        data = self._data(
            turns,
            {
                "timestamp_text": True,
                "target_t0": 10.0,
                "target_frames": 187,
                "prompt_frames": 40,
                "prev_frames": 10,
                "cond_frames": 50,
            },
        )
        out = preproc("uid", data)
        sp_id = preproc.token2id[SPEAKER_PROMPT_TOKEN]
        fill_id = preproc.token2id[TURN_FILL_TOKEN]
        for t in out["text"]:
            assert t.shape[0] == 50 + 187
            assert t[0].item() == sp_id
        assert (out["text"][0] == fill_id).sum().item() > 0

    def test_mode_o_unchanged(self, preproc):
        turns = [FakeTurn(channel=0, speaker="a", text="hi", start=10.5, end=11.5)]
        out = preproc("uid", self._data(turns))
        assert out["text"][0].shape[0] == 3  # <turn> + "h" + "i"
