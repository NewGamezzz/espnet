"""Branch-text preprocessing for multi-channel conversational TTS.

Implements the masking scheme fixed in PLAN-step2: branch ``i`` receives, for
each turn in conversation order, one ``<turn>`` marker followed by the turn's
characters if the turn belongs to channel ``i``, else exactly one ``<OTHER>``
token per character.  Turn markers carry no speaker identity, so no vocab
token depends on the speaker count.

This module is pure except for ``normalize_text``, which delegates to the
pretrained checkpoint's own tokenizer (``espnet2.text.f5_pinyin``, backed by
``rjieba``/``pypinyin``) so fine-tuning text matches the F5TTS_Base
pretraining distribution by construction.  No torch, no config access.
``turns`` arguments are duck-typed and only need ``channel`` and ``text``
attributes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

TURN_TOKEN = "<turn>"
OTHER_TOKEN = "<OTHER>"
NEW_TOKENS: tuple[str, str] = (TURN_TOKEN, OTHER_TOKEN)

# char_tokens.txt vocabs encode the space character as this symbol.
SPACE_SYMBOL = "<space>"


def extend_vocab(base_tokens: Sequence[str]) -> list[str]:
    """Append the two new tokens to the end of ``base_tokens``.

    Every base token keeps its id (line index); the new ids are contiguous at
    the end.  Raises ``ValueError`` if either token is already present, since
    that would silently break the "new token" assumption.
    """
    base = list(base_tokens)
    for token in NEW_TOKENS:
        if token in base:
            raise ValueError(f"token {token!r} already exists in the base vocab")
    return base + list(NEW_TOKENS)


def vocab_charset(tokens: Sequence[str]) -> frozenset[str]:
    """Characters representable by a vocab: single-char tokens, plus the space
    character when the vocab carries it as ``<space>``."""
    chars = {t for t in tokens if len(t) == 1}
    if SPACE_SYMBOL in tokens:
        chars.add(" ")
    return frozenset(chars)


def make_token2id(tokens: Sequence[str]) -> dict[str, int]:
    """Token -> id mapping (id = line index), aliasing ``" "`` to the
    ``<space>`` id when the vocab uses the symbol form.

    Raises ``ValueError`` on duplicate tokens: duplicate lines would make ids
    ambiguous.
    """
    token2id: dict[str, int] = {}
    for i, token in enumerate(tokens):
        if token in token2id:
            raise ValueError(
                f"duplicate token {token!r} at ids {token2id[token]} and {i}"
            )
        token2id[token] = i
    if " " not in token2id and SPACE_SYMBOL in token2id:
        token2id[" "] = token2id[SPACE_SYMBOL]
    return token2id


def normalize_text(text: str, charset: frozenset[str]) -> str:
    """Normalize a turn transcript against the vocab charset.

    The text first goes through the pretrained checkpoint's own tokenizer
    (``convert_char_to_pinyin``: ``;`` -> ``,`` and typographic-quote
    translations, plus jieba segmentation, which inserts a space after
    multi-letter segments so pretraining saw ``Turn-taking`` as
    ``Turn- taking``).  Feeding fine-tuning the same character distribution
    F5TTS_Base was trained on is the point; do NOT "clean up" these quirks.
    Every token that tokenizer returns must be a single character: a
    multi-char (pinyin) token means CJK input, which the one-``<OTHER>``-per-
    character budget cannot represent, so it fails loudly at build time.

    Per character afterwards: keep it if in ``charset``; else fall back to its
    lowercase form if that is in ``charset`` (lowercase-only vocabs must not
    corrupt words like "Can" -> "an"); else drop it, keeping whitespace as a
    word boundary.  Whitespace runs collapse to a single space and the result
    is stripped.  Runs once at build time so ``<OTHER>`` counts derived from
    the stored text can never desync between branches.
    """
    from espnet2.text.f5_pinyin import convert_char_to_pinyin

    tokens = convert_char_to_pinyin([text])
    for token in tokens[0]:
        if len(token) != 1:
            raise ValueError(
                f"F5 tokenizer produced non-character token {token!r} for "
                f"{text!r}; the char-level masking scheme supports "
                "English-only transcripts"
            )
    text = "".join(tokens[0])
    filtered = "".join(
        c if c in charset or c.isspace() else c.lower() if c.lower() in charset else ""
        for c in text
    )
    collapsed = " ".join(filtered.split())
    if " " not in charset:
        collapsed = collapsed.replace(" ", "")
    return collapsed


def build_branch_texts(turns: Sequence, num_channels: int) -> list[list[str]]:
    """Build per-branch token-string sequences from a turn list.

    Returns ``num_channels`` lists; branch ``i`` gets, per turn in the given
    order, ``[TURN_TOKEN]`` then the turn's characters if ``turn.channel == i``
    else ``len(turn.text)`` copies of ``OTHER_TOKEN``.  Token strings (not
    ids) so debug tools can render them without a vocab.
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    branches: list[list[str]] = [[] for _ in range(num_channels)]
    for turn in turns:
        if not 0 <= turn.channel < num_channels:
            raise ValueError(
                f"turn channel {turn.channel} out of range for {num_channels} channels"
            )
        for i, seq in enumerate(branches):
            seq.append(TURN_TOKEN)
            if i == turn.channel:
                seq.extend(turn.text)
            else:
                seq.extend([OTHER_TOKEN] * len(turn.text))
    return branches


def encode_tokens(tokens: Sequence[str], token2id: Mapping[str, int]) -> list[int]:
    """Map token strings to ids, failing loudly on any unknown token.

    After build-time normalization every token must be known; an OOV here is
    a pipeline bug, not user input to be cleaned.
    """
    ids = []
    for token in tokens:
        if token not in token2id:
            raise KeyError(f"token {token!r} not in vocab (size {len(token2id)})")
        ids.append(token2id[token])
    return ids


def render_tokens(tokens: Sequence[str]) -> str:
    """Human-readable one-line rendering: ``<turn>`` -> ``|``, ``<OTHER>`` -> ``#``."""
    return "".join(
        "|" if t == TURN_TOKEN else "#" if t == OTHER_TOKEN else t for t in tokens
    )
