"""Source classification and speaker/recording group ids for LEMAS keys.

The regexes are the audited ones from ``lemas_classify_key.py`` (vault note
"Findings - LEMAS Clean-Subset Key Audit", 2026-08-17). A key is
``<lang>_<source-specific id>``.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_RULES = [
    ("yodas", re.compile(r"^[A-Za-z0-9_-]{11}-\d{5}-\d{8}-\d{8}$")),
    ("mtedx", re.compile(r"^[A-Za-z0-9_-]{11}_\d{4}$")),
    ("mls", re.compile(r"^\d+_\d+_\d{6}$")),
    ("emilia", re.compile(r"^(EN_B\d{5}_S\d{5}_W\d{6}|emilia_zh_\d+)$")),
    ("wenetspeech4tts", re.compile(r"^WenetSpeech4TTS_\d+$")),
    ("gigaspeech2", re.compile(r"^train_\d+-\d+-\d+$")),
    ("gigaspeech", re.compile(r"^(POD|AUD|YOU)\d")),
    ("alcaim", re.compile(r"^[MF]\d{3}-\d{4}$")),
    ("golos", re.compile(r"^[0-9a-f]{32}$")),
]
SOURCES = [name for name, _ in _RULES] + ["unknown"]
_RECORDING_GROUP_SOURCES = {"yodas", "gigaspeech2", "mtedx"}
_EMILIA_EN = re.compile(r"^(EN_B\d{5}_S\d{5})_W\d{6}$")


def split_lang(key: str) -> Tuple[str, str]:
    """Split ``<lang>_<rest>``.

    Args:
        key: A LEMAS key, with or without the two-letter language prefix.

    Returns:
        ``(lang, rest)``; ``lang`` is ``""`` when there is no prefix.

    Example:
        >>> split_lang("de_9565_9808_002822")
        ('de', '9565_9808_002822')
    """
    if len(key) > 3 and key[2] == "_" and key[:2].isalpha() and key[:2].islower():
        return key[:2], key[3:]
    return "", key


def classify_key(key: str) -> str:
    """Map a key to its source corpus name.

    Args:
        key: A LEMAS key.

    Returns:
        One of ``SOURCES`` (``"unknown"`` when no rule matches).

    Example:
        >>> classify_key("de_9565_9808_002822")
        'mls'
    """
    _, rest = split_lang(key)
    for name, pattern in _RULES:
        if pattern.match(rest):
            return name
    return "unknown"


def group_id(key: str, source: str) -> Optional[str]:
    """Return the speaker or recording group id, or ``None``.

    Args:
        key: A LEMAS key.
        source: The key's source from :func:`classify_key`.

    Returns:
        A speaker id (MLS, Emilia EN, Alcaim), a recording id (YODAS,
        GigaSpeech2, mTEDx), or ``None`` for sources without structure
        (Emilia ZH, WenetSpeech4TTS, Golos, unknown).

    Example:
        >>> group_id("en_EN_B00012_S00913_W000420", "emilia")
        'EN_B00012_S00913'

    Note:
        A recording id is a speaker proxy, not a speaker: use
        :func:`is_recording_group` to tell the two apart.
    """
    _, rest = split_lang(key)
    if source in ("yodas", "mtedx"):
        return rest[:11]
    if source == "mls":
        return rest.split("_")[0]
    if source == "emilia":
        m = _EMILIA_EN.match(rest)
        return m.group(1) if m else None
    if source == "gigaspeech2":
        chan, vid, _seg = rest[len("train_") :].split("-")
        return f"{chan}-{vid}"
    if source == "alcaim":
        return rest.split("-")[0]
    return None


def segment_index(key: str, source: str) -> Optional[int]:
    """Return the within-recording segment index for recording-grouped sources.

    Args:
        key: A LEMAS key.
        source: The key's source from :func:`classify_key`.

    Returns:
        The segment index, or ``None`` when the source has no such notion.

    Example:
        >>> segment_index("de__2zsNO2V9K4-00556-00150066-00150154", "yodas")
        556
    """
    _, rest = split_lang(key)
    if source == "yodas":
        return int(rest[12:17])
    if source == "mtedx":
        return int(rest[12:])
    if source == "gigaspeech2":
        return int(rest.rsplit("-", 1)[1])
    return None


def is_recording_group(source: str) -> bool:
    """Whether ``source`` groups rows by recording rather than by speaker."""
    return source in _RECORDING_GROUP_SOURCES
