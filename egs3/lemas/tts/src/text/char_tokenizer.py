"""Stock F5-TTS text front-end behind the builder's ``phonemize`` interface.

The baseline row S0a trains F5-TTS as released: characters for every
language and pinyin syllables for Chinese, F5's ``Emilia_ZH_EN_pinyin``
scheme (``espnet3.systems.tts.f5_tts.pinyin.convert_char_to_pinyin``, which
runs ``rjieba`` on every input, English included). The only change is for
the manifest: F5's ``" "`` token, and any other whitespace character,
becomes ``<space>``, one per character (F5 maps every space token, so a run
of two stays two), because manifest column 3 is a space-separated token list
in a tab-separated file.
"""

from __future__ import annotations

from typing import List

from src.text.lemas_phonemizer import SPACE_TOKEN, LEMASPhonemizer

TEXT_UNITS = ("phones", "chars")


class F5CharTokenizer:
    """F5-TTS's character/pinyin tokenizer with manifest-safe tokens."""

    def __init__(self):
        """Import F5's tokenizer (``rjieba`` and ``pypinyin`` are needed).

        Example:
            >>> F5CharTokenizer().phonemize("hi du", "de")
            ['h', 'i', '<space>', 'd', 'u']
        """
        from espnet3.systems.tts.f5_tts.pinyin import convert_char_to_pinyin

        self._convert = convert_char_to_pinyin

    def phonemize(self, text: str, lang: str) -> List[str]:
        """Tokenize ``text`` as stock F5-TTS does; ``lang`` is not used.

        Args:
            text: Raw transcript.
            lang: Language code, accepted for interface parity with
                :class:`LEMASPhonemizer` (F5's tokenizer is language-blind).

        Returns:
            Characters, or pinyin syllables with tone numbers for hanzi, with
            ``<space>`` for every whitespace character; never an empty token.

        Raises:
            ValueError: If F5 emits a token with whitespace inside it, which
                the space-separated manifest column could not hold. ``|`` is
                kept as a character, as stock F5 does; it delimits only the
                ``split_targets`` column, which the builder guards itself.

        Example:
            >>> F5CharTokenizer().phonemize("中文", "zh")
            ['<space>', 'zhong1', '<space>', 'wen2']
        """
        out: List[str] = []
        for tok in self._convert([text])[0]:
            if not tok:
                continue
            if all(c.isspace() for c in tok):
                out.extend(SPACE_TOKEN for _ in tok)
                continue
            if any(c.isspace() for c in tok):
                raise ValueError(f"token {tok!r} cannot be stored in the manifest")
            out.append(tok)
        return out


def text_frontend(units: str):
    """Return the front-end for ``units``: ``phones`` or ``chars``.

    Args:
        units: One of :data:`TEXT_UNITS`.

    Returns:
        A :class:`LEMASPhonemizer` or an :class:`F5CharTokenizer`.

    Raises:
        ValueError: For any other value.

    Example:
        >>> type(text_frontend("chars")).__name__
        'F5CharTokenizer'
    """
    if units == "phones":
        return LEMASPhonemizer()
    if units == "chars":
        return F5CharTokenizer()
    raise ValueError(f"text_units must be one of {TEXT_UNITS}, got {units!r}")
