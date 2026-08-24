"""Tests for the branch-text masking scheme and vocab extension (AC1, AC2, AC9)."""

import pytest
from .conftest import FakeTurn

from egs3.conversational.tts.dataset.preprocessing.text import (
    FRAMES_PER_SECOND,
    NEW_TOKENS,
    OTHER_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
    build_branch_texts_timestamped,
    encode_tokens,
    extend_vocab,
    make_token2id,
    normalize_text,
    render_tokens,
    timestamp_fits,
    turn_frame_spans,
    vocab_charset,
)

N_SPK = 3


def _segments(branch: list[str]) -> list[list[str]]:
    """Split a branch token sequence at each TURN_TOKEN (one segment per turn)."""
    assert branch[0] == TURN_TOKEN
    segments: list[list[str]] = []
    for token in branch:
        if token == TURN_TOKEN:
            segments.append([])
        else:
            segments[-1].append(token)
    return segments


class TestMaskingRoundTrip:
    """AC1: own turns verbatim, exact <OTHER> counts, one <turn> per turn."""

    def test_one_turn_marker_per_turn_on_every_branch(self, turns_3spk):
        branches = build_branch_texts(turns_3spk, N_SPK)
        for branch in branches:
            assert branch.count(TURN_TOKEN) == len(turns_3spk)
            assert len(_segments(branch)) == len(turns_3spk)

    def test_own_turns_verbatim_and_in_order(self, turns_3spk):
        branches = build_branch_texts(turns_3spk, N_SPK)
        for i in range(N_SPK):
            segments = _segments(branches[i])
            own = [
                "".join(seg)
                for turn, seg in zip(turns_3spk, segments)
                if turn.channel == i
            ]
            expected = [turn.text for turn in turns_3spk if turn.channel == i]
            assert own == expected

    def test_other_turns_are_exact_other_runs(self, turns_3spk):
        branches = build_branch_texts(turns_3spk, N_SPK)
        for i in range(N_SPK):
            segments = _segments(branches[i])
            for turn, seg in zip(turns_3spk, segments):
                if turn.channel != i:
                    assert seg == [OTHER_TOKEN] * len(turn.text)

    def test_all_branches_same_length(self, turns_3spk):
        branches = build_branch_texts(turns_3spk, N_SPK)
        lengths = {len(b) for b in branches}
        assert len(lengths) == 1

    def test_channel_out_of_range_raises(self, turns_3spk):
        with pytest.raises(ValueError, match="out of range"):
            build_branch_texts(turns_3spk, 2)  # fixture uses channel 2


class TestNoLeak:
    """AC2: no character id of another speaker's turn appears in branch i."""

    def test_hidden_positions_are_only_other_ids(self, turns_3spk, base_vocab):
        token2id = make_token2id(extend_vocab(base_vocab))
        other_id = token2id[OTHER_TOKEN]
        turn_id = token2id[TURN_TOKEN]
        branches = build_branch_texts(turns_3spk, N_SPK)
        for i in range(N_SPK):
            ids = encode_tokens(branches[i], token2id)
            segments = _segments(branches[i])
            pos = 0
            for turn, seg in zip(turns_3spk, segments):
                assert ids[pos] == turn_id
                pos += 1
                seg_ids = ids[pos : pos + len(seg)]
                if turn.channel != i:
                    assert set(seg_ids) == {other_id}
                    hidden_ids = {token2id[c] for c in turn.text}
                    assert not hidden_ids & set(seg_ids)
                pos += len(seg)
            assert pos == len(ids)


class TestVocabExtension:
    """AC9: base ids unchanged, new ids contiguous at the end."""

    def test_base_ids_preserved_and_new_ids_at_end(self, base_vocab):
        extended = extend_vocab(base_vocab)
        assert extended[: len(base_vocab)] == base_vocab
        assert extended[len(base_vocab) :] == list(NEW_TOKENS)

    def test_existing_new_token_raises(self, base_vocab):
        with pytest.raises(ValueError, match="already exists"):
            extend_vocab(base_vocab + [OTHER_TOKEN])
        with pytest.raises(ValueError, match="already exists"):
            extend_vocab([TURN_TOKEN] + base_vocab)

    def test_make_token2id_space_alias_and_duplicates(self, base_vocab):
        token2id = make_token2id(base_vocab)
        assert token2id[" "] == token2id["<space>"]
        with pytest.raises(ValueError, match="duplicate"):
            make_token2id(base_vocab + ["a"])

    def test_charset_contains_space_via_symbol(self, base_vocab):
        charset = vocab_charset(base_vocab)
        assert " " in charset
        assert "a" in charset and "?" in charset
        assert "<blank>" not in charset

    def test_extend_vocab_appends_four_tokens_in_order(self):
        base = ["a", "b", "<space>"]
        out = extend_vocab(base)
        assert out[:3] == base
        assert out[3:] == ["<turn>", "<OTHER>", "<speaker_prompt>", "<prev_chunk>"]


class TestNormalizeAndEncode:
    def test_whitespace_collapse_and_oov_drop(self, base_vocab):
        charset = vocab_charset(extend_vocab(base_vocab))
        assert normalize_text("  can\tyou  hear\nme?  ", charset) == "can you hear me?"
        # OOV char between words keeps the word boundary.
        assert normalize_text("a ✓ b", charset) == "a b"
        assert normalize_text("café", charset) == "caf"

    def test_uppercase_falls_back_to_lowercase(self, base_vocab):
        charset = vocab_charset(extend_vocab(base_vocab))
        assert normalize_text("Can you hear Me?", charset) == "can you hear me?"

    def test_normalized_branches_stay_length_consistent(self, base_vocab):
        charset = vocab_charset(extend_vocab(base_vocab))
        turns = [
            FakeTurn(0, "s0", normalize_text("Heéllo   there", charset), 0.0, 1.0),
            FakeTurn(1, "s1", normalize_text("ok✓ay", charset), 1.5, 2.0),
        ]
        branches = build_branch_texts(turns, 2)
        assert len(branches[0]) == len(branches[1])

    def test_encode_strict_on_oov(self, base_vocab):
        token2id = make_token2id(extend_vocab(base_vocab))
        with pytest.raises(KeyError, match="not in vocab"):
            encode_tokens(["a", "é"], token2id)

    def test_render_tokens(self, turns_3spk):
        branches = build_branch_texts(turns_3spk[:2], 2)
        rendered = render_tokens(branches[0])
        assert rendered == "|good afternoon. how are you?|" + "#" * len(
            turns_3spk[1].text
        )

    def test_render_tokens_turn_fill(self):
        assert render_tokens([TURN_TOKEN, "a", TURN_FILL_TOKEN]) == "|a_"


def test_timestamp_generation_constants_and_extend_vocab_parity():
    # The 5-token generation is NEW_TOKENS plus <turn_fill> (timestamp
    # PR #42's vocab); extend_vocab must STILL append only NEW_TOKENS so
    # every eval fixture and golden stays byte-stable - the eval recipe
    # never builds a 5-token vocab, it only loads one.
    from egs3.conversational.tts.dataset.preprocessing.text import (
        TIMESTAMP_NEW_TOKENS,
        TURN_FILL_TOKEN,
    )

    assert TIMESTAMP_NEW_TOKENS == (*NEW_TOKENS, TURN_FILL_TOKEN)
    assert TURN_FILL_TOKEN == "<turn_fill>"
    assert extend_vocab(["a", "b", "<space>"])[3:] == list(NEW_TOKENS)


def _turn(channel, text, start, end):
    return FakeTurn(channel, f"spk_{channel}", text, start, end)


class TestTimestampAssembly:
    def test_single_turn_layout(self):
        # 2.0 s target = 187 frames at 93.75 fps; one turn [0.5, 1.5) = frames 47..141.
        turns = [_turn(0, "hi", 10.5, 11.5)]
        out = build_branch_texts_timestamped(turns, 2, t0=10.0, target_frames=187)
        a, b = out
        assert len(a) == len(b) == 187
        assert a[47] == TURN_TOKEN
        assert a[48:50] == ["h", "i"]
        assert a[50:141] == [TURN_FILL_TOKEN] * 91
        assert a[:47] == [OTHER_TOKEN] * 47 and a[141:] == [OTHER_TOKEN] * 46
        assert b == [OTHER_TOKEN] * 187

    def test_overlapping_turns_are_independent_per_branch(self):
        turns = [_turn(0, "abc", 0.0, 1.0), _turn(1, "de", 0.5, 1.5)]
        out = build_branch_texts_timestamped(turns, 2, t0=0.0, target_frames=187)
        assert out[0][0] == TURN_TOKEN and out[1][47] == TURN_TOKEN

    def test_unfittable_turn_raises(self):
        turns = [_turn(0, "way too much text", 0.0, 0.05)]
        with pytest.raises(ValueError, match="does not fit"):
            build_branch_texts_timestamped(turns, 1, t0=0.0, target_frames=187)

    def test_same_channel_collision_raises(self):
        turns = [_turn(0, "ab", 0.0, 1.0), _turn(0, "cd", 0.5, 1.5)]
        with pytest.raises(ValueError, match="overlap on channel"):
            build_branch_texts_timestamped(turns, 1, t0=0.0, target_frames=187)

    def test_timestamp_fits_normal_and_defective(self):
        good = [_turn(0, "hello there", 0.0, 2.0)]
        bad = [_turn(0, "I'm gonna go.", 0.0, 0.01)]
        assert timestamp_fits(good, t0=0.0, t1=2.0)
        assert not timestamp_fits(bad, t0=0.0, t1=2.0)

    def test_spans_clamped_to_target(self):
        turns = [_turn(0, "x", 0.0, 99.0)]
        assert turn_frame_spans(turns, t0=0.0, target_frames=100) == [(0, 100)]

    def test_frames_per_second_is_hop_ratio(self):
        assert FRAMES_PER_SECOND == 24000 / 256
