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
``turns`` arguments are duck-typed and only need ``channel``, ``text``,
``start``, and ``end`` attributes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

TURN_TOKEN = "<turn>"
OTHER_TOKEN = "<OTHER>"
SPEAKER_PROMPT_TOKEN = "<speaker_prompt>"
PREV_CHUNK_TOKEN = "<prev_chunk>"
# Mode T in-turn fill token from the timestamp-alignment training run
# (PR #42): pads a timestamp-aligned turn block to its span end.  Emitted by
# ``build_branch_texts_timestamped`` under ``text_format: timestamps``; it is
# the fifth (timestamp-era) vocab token, so the load gates must know it.
TURN_FILL_TOKEN = "<turn_fill>"
LEGACY_NEW_TOKENS: tuple[str, str] = (TURN_TOKEN, OTHER_TOKEN)
NEW_TOKENS: tuple[str, str, str, str] = (
    TURN_TOKEN,
    OTHER_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    PREV_CHUNK_TOKEN,
)
TIMESTAMP_NEW_TOKENS: tuple[str, str, str, str, str] = (*NEW_TOKENS, TURN_FILL_TOKEN)

# char_tokens.txt vocabs encode the space character as this symbol.
SPACE_SYMBOL = "<space>"


def extend_vocab(base_tokens: Sequence[str]) -> list[str]:
    """Append the new tokens to the end of ``base_tokens``.

    Every base token keeps its id (line index); the new ids are contiguous at
    the end.  Raises ``ValueError`` if any of them is already present, since
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
    """Human-readable one-line rendering: ``<turn>`` -> ``|``, ``<OTHER>`` ->
    ``#``, ``<turn_fill>`` -> ``_``."""
    return "".join(
        (
            "|"
            if t == TURN_TOKEN
            else "#" if t == OTHER_TOKEN else "_" if t == TURN_FILL_TOKEN else t
        )
        for t in tokens
    )


# Mel frame rate the Mode T frame grid is built on (24 kHz / hop 256).  The
# dataset asserts its fs/hop matches this whenever Mode T is enabled; keeping
# the constant here makes the pure assembly functions importable without any
# dataset config.
FRAMES_PER_SECOND: float = 93.75


def turn_frame_spans(
    turns: Sequence, t0: float, target_frames: int, fps: float = FRAMES_PER_SECOND
) -> list[tuple[int, int]]:
    """Per-turn ``[start_frame, end_frame)`` spans relative to the target
    start, rounded to the frame grid. Start is clamped to 0 (lower bound only);
    end is clamped to target_frames (upper bound only). A turn entirely before
    the target yields (0, -N); one entirely after yields (M, target_frames).
    Callers must treat non-positive-length spans as unfittable."""
    spans = []
    for turn in turns:
        s = int(round((turn.start - t0) * fps))
        e = int(round((turn.end - t0) * fps))
        spans.append((max(s, 0), min(e, target_frames)))
    return spans


def timestamp_fits(
    turns: Sequence, t0: float, t1: float, fps: float = FRAMES_PER_SECOND
) -> bool:
    """True iff every turn's ``<turn>`` + chars fits its frame span.

    Planner-side predicate for the Mode T coin: runs on seconds only, before
    any audio is read.  Uses ``floor((t1 - t0) * fps) - 1`` as the clamp
    bound - one frame of safety, because the dataset's actual frame count
    (``samples // hop``, after edge rounding and 48->24 kHz resampling) can
    sit one frame either side of the seconds estimate.  A pass here therefore
    guarantees ``build_branch_texts_timestamped`` succeeds at assembly time.
    Also rejects same-channel spans that collide after rounding (cannot
    happen for merge_gap-merged turns, but the predicate must imply assembly
    success, so it checks what assembly checks).
    """
    target = int((t1 - t0) * fps) - 1
    if target <= 0:
        return False
    last_end: dict[int, int] = {}
    for turn, (s, e) in zip(turns, turn_frame_spans(turns, t0, target, fps)):
        if e - s < 1 + len(turn.text):
            return False
        if last_end.get(turn.channel, 0) > s:
            return False
        last_end[turn.channel] = e
    return True


def build_branch_texts_timestamped(
    turns: Sequence,
    num_channels: int,
    t0: float,
    target_frames: int,
    fps: float = FRAMES_PER_SECOND,
) -> list[list[str]]:
    """Mode T branch texts: one token per mel frame over the target span.

    Branch ``i`` starts as ``target_frames`` copies of ``OTHER_TOKEN``; each
    of its own turns overwrites its frame span with ``TURN_TOKEN``, the
    turn's characters, then ``TURN_FILL_TOKEN`` to the span end.  Raises
    ``ValueError`` on a block that does not fit or on same-channel span
    collisions - callers gate windows with ``timestamp_fits`` first, so a
    raise here is a pipeline bug, not data to be cleaned (the
    ``encode_tokens`` convention).
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    branches = [[OTHER_TOKEN] * target_frames for _ in range(num_channels)]
    last_end = [0] * num_channels
    for turn, (s, e) in zip(turns, turn_frame_spans(turns, t0, target_frames, fps)):
        if not 0 <= turn.channel < num_channels:
            raise ValueError(
                f"turn channel {turn.channel} out of range for {num_channels} channels"
            )
        block = [TURN_TOKEN] + list(turn.text)
        if e - s < len(block):
            raise ValueError(
                f"turn block does not fit: needs {len(block)} frames, span "
                f"[{s}, {e}) has {e - s} (turn {turn.start:.3f}-{turn.end:.3f})"
            )
        if s < last_end[turn.channel]:
            raise ValueError(
                f"rounded turn spans overlap on channel {turn.channel} at frame {s}"
            )
        seq = branches[turn.channel]
        seq[s : s + len(block)] = block  # noqa: E203
        seq[s + len(block) : e] = [TURN_FILL_TOKEN] * (e - s - len(block))  # noqa: E203
        last_end[turn.channel] = e
    return branches
