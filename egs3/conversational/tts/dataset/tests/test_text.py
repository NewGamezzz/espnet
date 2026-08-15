"""Tests for the branch-text masking scheme and vocab extension (AC1, AC2, AC9)."""

import pytest
from .conftest import FakeTurn

from egs3.conversational.tts.dataset.preprocessing.text import (
    NEW_TOKENS,
    OTHER_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
    encode_tokens,
    extend_vocab,
    make_token2id,
    normalize_text,
    render_tokens,
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

    def test_new_tokens_is_five_tuple_ending_with_turn_fill(self):
        assert len(NEW_TOKENS) == 5
        assert NEW_TOKENS[-1] == TURN_FILL_TOKEN
        assert TURN_FILL_TOKEN == "<turn_fill>"

    def test_extend_vocab_appends_turn_fill_last(self):
        base = ["a", "b", "<space>"]
        extended = extend_vocab(base)
        assert extended[:3] == base
        assert extended[3:] == list(NEW_TOKENS)

    def test_new_token_constants(self):
        from egs3.conversational.tts.dataset.preprocessing.text import (
            NEW_TOKENS,
            SPEAKER_PROMPT_TOKEN,
            PREV_CHUNK_TOKEN,
        )

        assert NEW_TOKENS == (
            "<turn>",
            "<OTHER>",
            SPEAKER_PROMPT_TOKEN,
            PREV_CHUNK_TOKEN,
            TURN_FILL_TOKEN,
        )

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

    def test_render_tokens_handles_turn_fill(self):
        tokens = [TURN_TOKEN, "h", "i", TURN_FILL_TOKEN, OTHER_TOKEN]
        rendered = render_tokens(tokens)
        assert rendered == "|hi_#"


def _turn(channel, text, start, end):
    return FakeTurn(
        channel=channel, speaker=f"spk{channel}", text=text, start=start, end=end
    )


class TestTimestampAssembly:
    def test_single_turn_layout(self):
        # 2.0 s target = 187 frames at 93.75 fps; one turn [0.5, 1.5) = frames 47..141.
        turns = [_turn(0, "hi", 10.5, 11.5)]
        from egs3.conversational.tts.dataset.preprocessing.text import (
            build_branch_texts_timestamped,
        )

        out = build_branch_texts_timestamped(turns, 2, t0=10.0, target_frames=187)
        a, b = out
        assert len(a) == len(b) == 187
        assert a[47] == TURN_TOKEN
        assert a[48:50] == ["h", "i"]
        assert a[50:141] == [TURN_FILL_TOKEN] * 91
        assert a[:47] == [OTHER_TOKEN] * 47 and a[141:] == [OTHER_TOKEN] * 46
        assert b == [OTHER_TOKEN] * 187  # non-owner branch is all <OTHER>

    def test_overlapping_turns_are_independent_per_branch(self):
        from egs3.conversational.tts.dataset.preprocessing.text import (
            build_branch_texts_timestamped,
        )

        turns = [_turn(0, "abc", 0.0, 1.0), _turn(1, "de", 0.5, 1.5)]
        out = build_branch_texts_timestamped(turns, 2, t0=0.0, target_frames=187)
        assert (
            out[0][0] == TURN_TOKEN and out[1][47] == TURN_TOKEN
        )  # round(0.5 * 93.75) = 47

    def test_unfittable_turn_raises(self):
        from egs3.conversational.tts.dataset.preprocessing.text import (
            build_branch_texts_timestamped,
        )

        turns = [_turn(0, "way too much text", 0.0, 0.05)]  # 4 frames, needs 18
        with pytest.raises(ValueError, match="does not fit"):
            build_branch_texts_timestamped(turns, 1, t0=0.0, target_frames=187)

    def test_timestamp_fits_normal_and_defective(self):
        from egs3.conversational.tts.dataset.preprocessing.text import (
            timestamp_fits,
        )

        good = [_turn(0, "hello there", 0.0, 2.0)]
        bad = [_turn(0, "I'm gonna go.", 0.0, 0.01)]
        assert timestamp_fits(good, t0=0.0, t1=2.0)
        assert not timestamp_fits(bad, t0=0.0, t1=2.0)

    def test_spans_clamped_to_target(self):
        from egs3.conversational.tts.dataset.preprocessing.text import (
            turn_frame_spans,
        )

        turns = [_turn(0, "x", 0.0, 99.0)]
        spans = turn_frame_spans(turns, t0=0.0, target_frames=100)
        assert spans == [(0, 100)]
