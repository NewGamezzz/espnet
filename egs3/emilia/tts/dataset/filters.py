"""Upstream-faithful Emilia text and duration filters.

Ported from SWivid/F5-TTS ``src/f5_tts/train/datasets/prepare_emilia.py``.
Upstream matches the blocklists against the record's ``speaker`` field; the
staged prep on PSC matched ``id`` instead, which carries an extra ``_W``
window field, so its blocklists never fired (spec section 2.2).
"""

from __future__ import annotations

import re
from collections import Counter

OUT_ZH = frozenset(
    {
        "ZH_B00041_S06226",
        "ZH_B00042_S09204",
        "ZH_B00065_S09430",
        "ZH_B00065_S09431",
        "ZH_B00066_S09327",
        "ZH_B00066_S09328",
    }
)

OUT_EN = frozenset(
    {
        "EN_B00013_S00913",
        "EN_B00042_S00120",
        "EN_B00055_S04111",
        "EN_B00061_S00693",
        "EN_B00061_S01494",
        "EN_B00061_S03375",
        "EN_B00059_S00092",
        "EN_B00111_S04300",
        "EN_B00100_S03759",
        "EN_B00087_S03811",
        "EN_B00059_S00950",
        "EN_B00089_S00946",
        "EN_B00078_S05127",
        "EN_B00070_S04089",
        "EN_B00074_S09659",
        "EN_B00061_S06983",
        "EN_B00061_S07060",
        "EN_B00059_S08397",
        "EN_B00082_S06192",
        "EN_B00091_S01238",
        "EN_B00089_S07349",
        "EN_B00070_S04343",
        "EN_B00061_S02400",
        "EN_B00076_S01262",
        "EN_B00068_S06467",
        "EN_B00076_S02943",
        "EN_B00064_S05954",
        "EN_B00061_S05386",
        "EN_B00066_S06544",
        "EN_B00076_S06944",
        "EN_B00072_S08620",
        "EN_B00076_S07135",
        "EN_B00076_S09127",
        "EN_B00065_S00497",
        "EN_B00059_S06227",
        "EN_B00063_S02859",
        "EN_B00075_S01547",
        "EN_B00061_S08286",
        "EN_B00079_S02901",
        "EN_B00092_S03643",
        "EN_B00096_S08653",
        "EN_B00063_S04297",
        "EN_B00063_S04614",
        "EN_B00079_S04698",
        "EN_B00104_S01666",
        "EN_B00061_S09504",
        "EN_B00061_S09694",
        "EN_B00065_S05444",
        "EN_B00063_S06860",
        "EN_B00065_S05725",
        "EN_B00069_S07628",
        "EN_B00083_S03875",
        "EN_B00071_S07665",
        "EN_B00062_S04187",
        "EN_B00065_S09873",
        "EN_B00065_S09922",
        "EN_B00084_S02463",
        "EN_B00067_S05066",
        "EN_B00106_S08060",
        "EN_B00073_S06399",
        "EN_B00073_S09236",
        "EN_B00087_S00432",
        "EN_B00085_S05618",
        "EN_B00064_S01262",
        "EN_B00072_S01739",
        "EN_B00059_S03913",
        "EN_B00069_S04036",
        "EN_B00067_S05623",
        "EN_B00060_S05389",
        "EN_B00060_S07290",
        "EN_B00062_S08995",
    }
)

ZH_CHAR_FILTERS = ("い", "て")
EN_CHAR_FILTERS = ("ا", "い", "て")

ZH_PUNCT_TABLE = str.maketrans({",": "，", "!": "！", "?": "？"})

# dataset/builder.py writes normalize_text's output as the last, unquoted
# field of a tab-separated manifest row (IMPORTANT: TSV, minor #4 in the
# final whole-branch review). A literal '\n', '\r' or '\t' inside a JSON
# text field would otherwise split or shift that row.
_ROW_BREAKING_WHITESPACE_TABLE = str.maketrans({"\n": " ", "\r": " ", "\t": " "})

_REPEAT_RUN = re.compile(r"(.)\1{10,}")
_LONG_DIGITS = re.compile(r"\d{15,}")

# IMPORTANT 4 (final whole-branch review): keep_utterance used to dispatch
# these three per-language rules with `X if lang == "EN" else Y`, which
# silently applies the ZH rule to ANY lang that isn't the literal string
# "EN" -- including a typo, or a genuinely new third language, since
# dataset/config.yaml exposes `langs` as a config knob with no validation
# against these tables. Dict lookups keyed by language raise on an unknown
# key instead of silently guessing.
_BLOCKLIST_BY_LANG = {"EN": OUT_EN, "ZH": OUT_ZH}
_CHAR_FILTERS_BY_LANG = {"EN": EN_CHAR_FILTERS, "ZH": ZH_CHAR_FILTERS}
_REPETITION_LENGTH_BY_LANG = {"EN": 4, "ZH": 2}


def _lookup_by_lang(mapping: dict, lang: str, what: str):
    """Look up a per-language filter table, raising on an unknown lang."""
    try:
        return mapping[lang]
    except KeyError:
        raise ValueError(
            f"keep_utterance: no {what} configured for lang={lang!r}. "
            f"Known languages: {sorted(mapping)}."
        ) from None


def repetition_found(text: str, length: int = 2, tolerance: int = 10) -> bool:
    """True if any ``length``-CHARACTER substring occurs more than ``tolerance`` times.

    Verbatim port of ``f5_tts.model.utils.repetition_found``. Both details
    below are load-bearing and an earlier version of this file got both
    wrong, so they are spelled out:

    - The n-grams are **characters**, sliced straight off the string, not
      words. Emilia's ZH text carries no spaces, so a word-based version
      degenerates to a single token and the filter becomes a silent no-op
      for the entire Chinese half of the corpus.
    - The threshold is ``count > tolerance`` with ``tolerance=10``, not
      "appears twice". A word-based, appears-twice version rejects ordinary
      English such as "I went to the store and I went to the store again",
      which upstream keeps.

    Callers pass ``length=4`` for EN and leave the default 2 for ZH, matching
    upstream's ``repetition_found(text, length=4)`` and ``repetition_found(text)``.
    """
    pattern_count: dict = {}
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
    for count in pattern_count.values():
        if count > tolerance:
            return True
    return False


def normalize_text(text: str, lang: str) -> str:
    """Strip, and for ZH map ASCII punctuation to full-width.

    Also replaces embedded newlines/carriage-returns/tabs with a single
    space each: dataset/builder.py writes this text as the last, unquoted
    field of a tab-separated manifest row with no escaping. A literal
    ``\\n``/``\\r``/``\\t`` inside a JSON ``text`` field would otherwise
    split or shift the row, silently corrupting the manifest -- a defect
    that would only show up once, at a random position, somewhere across
    37M rows.
    """
    text = text.strip()
    text = text.translate(_ROW_BREAKING_WHITESPACE_TABLE)
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

    blocklist = _lookup_by_lang(_BLOCKLIST_BY_LANG, lang, "blocklist")
    if speaker in blocklist:
        return False, "blocklist"

    char_filters = _lookup_by_lang(_CHAR_FILTERS_BY_LANG, lang, "char filters")
    if any(ch in text for ch in char_filters):
        return False, "charfilter"

    rep_len = _lookup_by_lang(_REPETITION_LENGTH_BY_LANG, lang, "repetition length")
    if repetition_found(text, length=rep_len):
        return False, "repetition"

    if not (min_duration <= duration <= max_duration):
        return False, "duration"

    if strict and _strict_reject(text):
        return False, "strict"

    return True, ""
