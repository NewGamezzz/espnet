"""Input-preprocessing parity with the pretrained F5TTS_Base conventions.

The conversational pipeline assumes that (a) raw character ids against the
Emilia vocab are exactly what the pretrained model saw in training, and
(b) the mel front-end frame accounting matches what the sampling scripts
assume.  These tests pin both assumptions:

* id-level parity between the conversational char encoding and F5's own
  ``text_to_pinyin_ids`` tokenizer on normalized English text (needs the
  downloaded base vocab + ``pypinyin``/``rjieba``),
* the one known, deliberate divergence (F5's ``;`` -> ``,`` translation),
* masked-script invariants on a constructed two-channel window,
* the ``T_wav // hop + 1`` frame-length convention and stability of a padded
  row's frames against its unpadded computation (the packed-collator case).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .conftest import REPO_ROOT  # noqa: F401  (side effect: repo root on sys.path)

from egs3.conversational.tts.dataset.preprocessing.text import (  # noqa: E402
    OTHER_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
    encode_tokens,
    make_token2id,
    normalize_text,
    render_tokens,
    vocab_charset,
)
from egs3.conversational.tts.dataset.preprocessor import read_vocab  # noqa: E402

RECIPE_DIR = Path(__file__).resolve().parents[1]
BASE_VOCAB = RECIPE_DIR / "downloads" / "F5TTS_Base" / "vocab.txt"

needs_vocab = pytest.mark.skipif(
    not BASE_VOCAB.exists(),
    reason="pretrained F5TTS_Base vocab not downloaded (see conf/training_poc.yaml)",
)


@needs_vocab
def test_char_ids_match_f5_pinyin_tokenizer():
    """Normalized text must encode to the exact ids the pretrained model saw.

    The conversational pipeline encodes raw characters (id = vocab line
    index); the pretrained F5TTS_Base was trained on ``convert_char_to_pinyin``
    output, which is char-identity for English.  Any drift here would feed
    the model well-formed but wrongly-mapped ids: it loads cleanly and
    degrades silently.
    """
    pytest.importorskip("pypinyin")
    pytest.importorskip("rjieba")
    from espnet2.text.f5_pinyin import load_vocab_char_map, text_to_pinyin_ids

    base = read_vocab(BASE_VOCAB)
    token2id = make_token2id(base)
    charset = vocab_charset(base)
    char_map = load_vocab_char_map(str(BASE_VOCAB))

    # Plain conversational text: no hyphenated compounds or semicolons (see
    # the known-divergence tests below for those).
    sentences = [
        "Some call me nature, others call me mother nature.",
        "The quick brown fox jumps over the lazy dog!",
        "hello world, how are you today?",
        "I mean, yeah. You know what? That's right.",
    ]
    for sentence in sentences:
        normalized = normalize_text(sentence, charset)
        assert normalized, sentence
        ours = encode_tokens(list(normalized), token2id)
        f5 = text_to_pinyin_ids(normalized, char_map).tolist()
        assert ours == f5, f"id divergence on {normalized!r}"


@needs_vocab
def test_semicolon_translation_is_a_known_divergence():
    """F5's tokenizer translates ``;`` to ``,`` before the id lookup; the
    conversational ``normalize_text`` keeps ``;`` whenever the vocab carries
    it.  This test documents the divergence: transcripts containing ``;``
    reach the model with an id the pretrained model essentially never saw.
    If SSSD transcripts contain semicolons, map them to commas at build time.
    """
    pytest.importorskip("pypinyin")
    pytest.importorskip("rjieba")
    from espnet2.text.f5_pinyin import load_vocab_char_map, text_to_pinyin_ids

    base = read_vocab(BASE_VOCAB)
    char_map = load_vocab_char_map(str(BASE_VOCAB))
    assert ";" in vocab_charset(base), "vocab dropped ';'; divergence moot"
    f5 = text_to_pinyin_ids("a;b", char_map).tolist()
    assert f5 == text_to_pinyin_ids("a,b", char_map).tolist()
    ours = encode_tokens(list("a;b"), make_token2id(base))
    assert ours != f5


@needs_vocab
def test_hyphen_compound_space_is_a_known_divergence():
    """F5's tokenizer segments text with jieba and inserts a space after a
    multi-letter segment, so pretraining saw hyphenated compounds as
    ``Turn- taking`` (extra space), while the conversational encoding keeps
    ``Turn-taking`` verbatim.  Single-letter fragments (``a-b``) are NOT
    affected.  Documented divergence: if SSSD transcripts are hyphen-heavy,
    consider replacing ``-`` with a space at build time.
    """
    pytest.importorskip("pypinyin")
    pytest.importorskip("rjieba")
    from espnet2.text.f5_pinyin import convert_char_to_pinyin

    assert convert_char_to_pinyin(["Turn-taking"])[0] == list("Turn- taking")
    assert convert_char_to_pinyin(["a-b"])[0] == list("a-b")


def test_masked_scripts_two_channel_window():
    """Per-turn structure of the masked scripts on a constructed window."""
    turns = [
        SimpleNamespace(channel=0, text="hello there"),
        SimpleNamespace(channel=1, text="hi"),
        SimpleNamespace(channel=0, text="how are you"),
    ]
    branches = build_branch_texts(turns, 2)

    # Same aligned length on both branches, one <turn> marker per turn.
    assert len(branches[0]) == len(branches[1])
    assert branches[0].count(TURN_TOKEN) == len(turns)
    assert branches[1].count(TURN_TOKEN) == len(turns)

    # Own turns verbatim, other turns exactly one <OTHER> per hidden char.
    assert branches[0] == (
        [TURN_TOKEN]
        + list("hello there")
        + [TURN_TOKEN]
        + [OTHER_TOKEN] * len("hi")
        + [TURN_TOKEN]
        + list("how are you")
    )
    assert branches[1] == (
        [TURN_TOKEN]
        + [OTHER_TOKEN] * len("hello there")
        + [TURN_TOKEN]
        + list("hi")
        + [TURN_TOKEN]
        + [OTHER_TOKEN] * len("how are you")
    )

    # No leakage: wherever a branch hides a turn, no base-vocab char appears.
    vocab = [" "] + sorted(set("helothrawyoui"))
    token2id = make_token2id(vocab + [TURN_TOKEN, OTHER_TOKEN])
    other_id = token2id[OTHER_TOKEN]
    ids1 = encode_tokens(branches[1], token2id)
    hidden = slice(1, 1 + len("hello there"))  # after the first <turn>
    assert all(i == other_id for i in ids1[hidden])

    assert render_tokens(branches[0]) == "|hello there|##|how are you"


def test_frame_length_convention_and_padded_row_stability():
    """The front-end's frame accounting must match what the recipe assumes.

    ``VocoderMelSpec`` documents ``T_wav // hop + 1`` (center=True STFT); the
    packed collator zero-pads rows to the longest window, so a padded row's
    frames must match its unpadded computation everywhere the analysis
    window does not touch the padding boundary.
    """
    from espnet2.tts.feats_extract.vocoder_mel import VocoderMelSpec

    fe = VocoderMelSpec(
        fs=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=100,
        mel_spec_type="vocos",
    )
    hop, n_fft = 256, 1024

    generator = torch.Generator().manual_seed(0)
    short = 0.1 * torch.randn(1, 24000 + 123, generator=generator)  # odd length
    long_ = 0.1 * torch.randn(1, 2 * 24000, generator=generator)

    # Packed batch: short row zero-padded to the long row's length.
    packed = torch.zeros(2, long_.shape[1])
    packed[0, : short.shape[1]] = short[0]
    packed[1] = long_[0]
    lengths = torch.tensor([short.shape[1], long_.shape[1]])
    feats, feats_lengths = fe(packed, lengths)

    assert feats_lengths.tolist() == [
        short.shape[1] // hop + 1,
        long_.shape[1] // hop + 1,
    ]
    assert feats.shape[1] == long_.shape[1] // hop + 1

    # Unpadded reference for the short row; frames whose window stays clear
    # of the padding boundary must be bit-identical.
    ref, ref_len = fe(short, torch.tensor([short.shape[1]]))
    stable = (short.shape[1] - n_fft // 2) // hop
    torch.testing.assert_close(feats[0, :stable], ref[0, :stable], rtol=0.0, atol=0.0)
