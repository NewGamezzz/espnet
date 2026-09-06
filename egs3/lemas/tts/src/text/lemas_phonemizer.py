"""Multilingual phoneme front-end built on espnet2's g2p modules.

LEMAS scheme (arXiv 2601.04233): pinyin initial-final with tones for zh via
``espnet2.text.phoneme_tokenizer.pypinyin_g2p_phone``; eSpeak-NG IPA for the
other languages via ``espnet2.text.phoneme_tokenizer.Phonemizer``. Tokens are
per phone, stress stays attached to its phone, punctuation (any Unicode
punctuation category) is kept as its own tokens, words are separated by
``<space>``. Non-IPA phone symbols espeak emits (``ə-``, ``e-``, ``ɪ^``) are
kept verbatim.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence

SPK_TOKEN = "<spk>"
LANG_TOKEN = "<lang>"
SPACE_TOKEN = "<space>"
LANGS = ["de", "en", "es", "fr", "id", "it", "pt", "ru", "vi", "zh"]
ESPEAK_VOICES = {
    "de": "de",
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
    "id": "id",
    "it": "it",
    "pt": "pt-br",
    "ru": "ru",
    "vi": "vi",
}
_PHONE_SEP = "|"
_WORD_SEP = " "
_DROP = {"-"}  # espeak liaison / hyphen marker, carries no phone


def lang_tag(lang: str) -> str:
    """Return the target-language tag token, e.g. ``<de>``."""
    return f"<{lang}>"


def special_tokens() -> List[str]:
    """Return the role tokens followed by one language tag per language.

    Returns:
        ``["<spk>", "<lang>", "<de>", ..., "<zh>"]``.

    Example:
        >>> special_tokens()[:3]
        ['<spk>', '<lang>', '<de>']
    """
    return [SPK_TOKEN, LANG_TOKEN] + [lang_tag(lang) for lang in LANGS]


def _is_punct(c: str) -> bool:
    # Any Unicode punctuation (Po, Pd, Ps, Pe, Pi, Pf, Pc) splits off as its own
    # token, so «», „“”, ¿¡, CJK 。，、 and an inner "." in "e.g." all separate.
    # A plain hyphen stays inside a phone: espeak uses it in phone symbols such
    # as the French elidable schwa "ə-" and the Vietnamese vowel "e-".
    return c != "-" and unicodedata.category(c).startswith("P")


def _split_punct(token: str) -> List[str]:
    out: List[str] = []
    cur = ""
    for c in token:
        if _is_punct(c):
            if cur:
                out.append(cur)
                cur = ""
            out.append(c)
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


class LEMASPhonemizer:
    """Per-language g2p dispatcher over espnet2's phoneme modules."""

    def __init__(self, langs: Sequence[str] = LANGS):
        """Build one g2p per language.

        Args:
            langs: Languages to support; each espeak language constructs a
                ``Phonemizer`` (which loads eSpeak-NG), zh uses pypinyin.

        Raises:
            KeyError: If a language has no espeak voice mapping.

        Example:
            >>> LEMASPhonemizer(langs=["zh"]).phonemize("你好", "zh")
            ['n', 'i3', 'h', 'ao3']

        Note:
            Construction is eager so that a missing eSpeak-NG library fails at
            start-up rather than inside a dataloader worker.
        """
        self._g2p: Dict[str, object] = {}
        for lang in langs:
            if lang == "zh":
                from espnet2.text.phoneme_tokenizer import pypinyin_g2p_phone

                self._g2p[lang] = pypinyin_g2p_phone
            else:
                from espnet2.text.phoneme_tokenizer import Phonemizer

                self._g2p[lang] = Phonemizer(
                    language=ESPEAK_VOICES[lang],
                    backend="espeak",
                    with_stress=True,
                    preserve_punctuation=True,
                    language_switch="remove-flags",
                    strip=True,
                    word_separator=_WORD_SEP,
                    phone_separator=_PHONE_SEP,
                )

    def phonemize(self, text: str, lang: str) -> List[str]:
        """Turn text into the token sequence stored in manifest column 3.

        Args:
            text: Raw transcript.
            lang: Two-letter language code.

        Returns:
            Phone tokens with ``<space>`` between words; punctuation kept as
            its own tokens.

        Raises:
            KeyError: If ``lang`` was not requested at construction.

        Example:
            >>> LEMASPhonemizer(langs=["zh"]).phonemize("你好", "zh")
            ['n', 'i3', 'h', 'ao3']
        """
        g2p = self._g2p[lang]
        if lang == "zh":
            return [t for t in g2p(text) if t.strip()]
        words = g2p(text)
        out: List[str] = []
        for i, word in enumerate(words):
            if i:
                out.append(SPACE_TOKEN)
            for phone in word.split(_PHONE_SEP):
                out.extend(p for p in _split_punct(phone) if p and p not in _DROP)
        return out

    def phonemize_words(self, words: Sequence[str], lang: str) -> List[List[str]]:
        """Phonemize each word on its own (used for split-mode rows)."""
        return [self.phonemize(w, lang) for w in words]
