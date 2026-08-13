"""Upstream-faithful Emilia text and duration filters.

Ported from SWivid/F5-TTS ``src/f5_tts/train/datasets/prepare_emilia.py``.
Upstream matches the blocklists against the record's ``speaker`` field; the
staged prep on PSC matched ``id`` instead, which carries an extra ``_W``
window field, so its blocklists never fired (spec section 2.2).
"""

from __future__ import annotations

import re
from collections import Counter

OUT_ZH = frozenset({
    "ZH_B00041_S06226", "ZH_B00042_S09204", "ZH_B00065_S09430",
    "ZH_B00065_S09431", "ZH_B00066_S09327", "ZH_B00066_S09328",
})

OUT_EN = frozenset({
    "EN_B00013_S00913", "EN_B00042_S00120", "EN_B00055_S04111",
    "EN_B00061_S00693", "EN_B00061_S01494", "EN_B00061_S03375",
    "EN_B00059_S00092", "EN_B00111_S04300", "EN_B00100_S03759",
    "EN_B00087_S03811", "EN_B00059_S00950", "EN_B00089_S00946",
    "EN_B00078_S05127", "EN_B00070_S04089", "EN_B00074_S09659",
    "EN_B00061_S06983", "EN_B00061_S07060", "EN_B00059_S08397",
    "EN_B00082_S06192", "EN_B00091_S01238", "EN_B00089_S07349",
    "EN_B00070_S04343", "EN_B00061_S02400", "EN_B00076_S01262",
    "EN_B00068_S06467", "EN_B00076_S02943", "EN_B00064_S05954",
    "EN_B00061_S05386", "EN_B00066_S06544", "EN_B00076_S06944",
    "EN_B00072_S08620", "EN_B00076_S07135", "EN_B00076_S09127",
    "EN_B00065_S00497", "EN_B00059_S06227", "EN_B00063_S02859",
    "EN_B00075_S01547", "EN_B00061_S08286", "EN_B00079_S02901",
    "EN_B00092_S03643", "EN_B00096_S08653", "EN_B00063_S04297",
    "EN_B00063_S04614", "EN_B00079_S04698", "EN_B00104_S01666",
    "EN_B00061_S09504", "EN_B00061_S09694", "EN_B00065_S05444",
    "EN_B00063_S06860", "EN_B00065_S05725", "EN_B00069_S07628",
    "EN_B00083_S03875", "EN_B00071_S07665", "EN_B00062_S04187",
    "EN_B00065_S09873", "EN_B00065_S09922", "EN_B00084_S02463",
    "EN_B00067_S05066", "EN_B00106_S08060", "EN_B00073_S06399",
    "EN_B00073_S09236", "EN_B00087_S00432", "EN_B00085_S05618",
    "EN_B00064_S01262", "EN_B00072_S01739", "EN_B00059_S03913",
    "EN_B00069_S04036", "EN_B00067_S05623", "EN_B00060_S05389",
    "EN_B00060_S07290", "EN_B00062_S08995",
})

ZH_CHAR_FILTERS = ("い", "て")
EN_CHAR_FILTERS = ("ا", "い", "て")

ZH_PUNCT_TABLE = str.maketrans({",": "，", "!": "！", "?": "？"})

_REPEAT_RUN = re.compile(r"(.)\1{10,}")
_LONG_DIGITS = re.compile(r"\d{15,}")


def repetition_found(text: str, length: int = 2) -> bool:
    """True if any ``length``-word n-gram occurs more than once."""
    words = text.split()
    if len(words) < length:
        return False
    seen: set = set()
    for i in range(len(words) - length + 1):
        ngram = tuple(words[i:i + length])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def normalize_text(text: str, lang: str) -> str:
    """Strip, and for ZH map ASCII punctuation to full-width."""
    text = text.strip()
    if lang == "ZH":
        text = text.translate(ZH_PUNCT_TABLE)
    return text


def _strict_reject(text: str) -> bool:
    """filter_text.py's extra rules. Off unless strict=True (D3)."""
    if _REPEAT_RUN.search(text):
        return True
    if _LONG_DIGITS.search(text):
        return True
    compact = "".join(text.lower().split())
    if not compact:
        return True
    most_common = Counter(compact).most_common(1)[0][1]
    return most_common / len(compact) > 0.5


def keep_utterance(
    record: dict,
    lang: str,
    min_duration: float,
    max_duration: float,
    strict: bool = False,
) -> tuple[bool, str]:
    """Decide whether one Emilia JSON record survives filtering.

    Returns ``(keep, reason)``; ``reason`` is ``""`` when kept.
    """
    speaker = record["speaker"]
    text = record["text"].strip()
    duration = float(record["duration"])

    blocklist = OUT_EN if lang == "EN" else OUT_ZH
    if speaker in blocklist:
        return False, "blocklist"

    char_filters = EN_CHAR_FILTERS if lang == "EN" else ZH_CHAR_FILTERS
    if any(ch in text for ch in char_filters):
        return False, "charfilter"

    rep_len = 4 if lang == "EN" else 2
    if repetition_found(text, length=rep_len):
        return False, "repetition"

    if not (min_duration <= duration <= max_duration):
        return False, "duration"

    if strict and _strict_reject(text):
        return False, "strict"

    return True, ""
