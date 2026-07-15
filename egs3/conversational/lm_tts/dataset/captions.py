"""Narration-style caption and chain-of-thought templates (Task 4).

Pure text module: no I/O beyond ``apply_paraphrase_overlay`` reading one JSON
file, no torch. Consumes ``dataset.preprocessing.attributes.SpeakerAttrs``
(Task 3) and ``dataset.preprocessing.sssd.Turn`` and renders the SFT caption
style documented in ``docs/bagpiper-findings.md`` ("SFT data schema" section,
especially caption example 5 and the complete verbatim
``dev_multi_talker`` entry): a labeled voice-description block, a blank line,
then labeled quoted script lines.

Quote convention: the real SFT captions mix straight (``"``) and curly
(``“``/``”``) quotation marks inconsistently (compare caption example 5,
which uses straight quotes, against the verbatim multi-talker entry, which
uses curly quotes around the same ``"<speaker> says:"`` pattern). Every
template in this module standardizes on the plain straight double-quote
character for the quoted script/turn text.

Literal double quotes inside a script/turn (e.g. a speaker quoting someone
else: ``She said "hello" to me.``) pass through into the wrapping quotes
unescaped. This is deliberate: verbatim speech is never mangled or
re-escaped by this module.

DESIGN-CRITICAL (project decision 14): a capability probe showed the
checkpoint's audio decoder degenerates on persona/character-framed captions
("play the role of...", "embody the character of...") while narration-style
captions ("A female speaker with a ... voice ... says: ...") work. Every
piece of text this module *generates* - ``voice_description`` output and
``cot_block`` output, plus paraphrase overlay entries - is guarded by
``assert_no_persona``. Verbatim script/turn text (the actual words a speaker
says) is deliberately NOT scanned: real conversational transcripts routinely
contain "role" or "character" as ordinary words (e.g. "she played a role in
the decision", "he's such a character"), and blanket-scanning the assembled
caption would raise on innocent transcripts that have nothing to do with
persona framing. The regression surface this guard protects is the template
vocabulary and any paraphrase overlay, i.e. the parts of the caption this
codebase writes, not the parts a real speaker said.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .preprocessing.attributes import SpeakerAttrs
from .preprocessing.sssd import Turn

# Band -> caption vocabulary. One fixed rendering per band, deterministic,
# vocabulary drawn from the SFT caption register (bright/warm/calm/clear).
# DESIGN-CRITICAL: keyed by the exact band strings produced by
# dataset.preprocessing.attributes (pitch_band/variability_band/rate_band);
# do not change those bands without updating this table.
PITCH_ADJ = {
    "low": "deep",
    "medium": "medium",
    "high": "high",
}

VARIABILITY_ADJ = {
    "flat": "calm, even",
    "expressive": "lively, expressive",
}

RATE_PHRASE = {
    "measured": "an unhurried pace",
    "moderate": "a natural pace",
    "brisk": "a quick pace",
}

# Persona/character-framing regression guard (decision 14). Word-boundary
# anchored so substrings inside ordinary words ("personality",
# "characteristic") are not false positives. Covers common inflections
# (roles/characters/personas, embody/embodies/embodied/embodying/embodiment,
# portray/portrays/portrayed/portraying/portrayal, pretend/pretends/
# pretended/pretending, role-play/roleplay/role-plays/roleplaying/
# role-played) while staying word-boundary anchored.
PERSONA_MARKERS = re.compile(
    r"\b("
    r"roles?|characters?|personas?"
    r"|embod(?:y|ies|ied|ying|iment)"
    r"|portray(?:s|ed|ing|al)?"
    r"|pretend(?:s|ed|ing)?"
    r"|role-?play(?:s|ing|ed)?"
    r")\b",
    re.IGNORECASE,
)


def assert_no_persona(text: str) -> None:
    """Raise ``ValueError`` if ``text`` contains a persona/character marker.

    See the module docstring for what this is (and is not) applied to.
    """
    match = PERSONA_MARKERS.search(text)
    if match:
        raise ValueError(
            f"persona/character marker {match.group(0)!r} found in generated "
            "caption text; narration-style captions only (decision 14): "
            f"{text!r}"
        )


def _strip_leading_article(description: str) -> str:
    """Drop a leading "A "/"An " so ``description`` can follow a label/colon."""
    if description.startswith("An "):
        return description[3:]
    if description.startswith("A "):
        return description[2:]
    return description


def _voice_clause(attrs: SpeakerAttrs) -> str:
    """Voice description with no leading article and no trailing period, for
    splicing into a larger sentence (used by ``cot_block``)."""
    return _strip_leading_article(voice_description(attrs)).rstrip(".")


def voice_description(attrs: SpeakerAttrs) -> str:
    """One-sentence voice description plus a recording-quality clause.

    Template: "A {gender} speaker with a {pitch}-pitched, {variability}
    voice, speaking at {rate} pace in a clean close-microphone recording."
    """
    pitch_adj = PITCH_ADJ[attrs.pitch_band]
    variability_adj = VARIABILITY_ADJ[attrs.variability_band]
    rate_phrase = RATE_PHRASE[attrs.rate_band]
    text = (
        f"A {attrs.gender} speaker with a {pitch_adj}-pitched, "
        f"{variability_adj} voice, speaking at {rate_phrase} in a clean "
        "close-microphone recording."
    )
    assert_no_persona(text)
    return text


def setting_sentence() -> str:
    """Fixed TAC setting sentence: zero partner content, one side of a
    two-person conversation."""
    return (
        "The following is one side of a natural two-person conversation; "
        "the speaker responds in their own turns."
    )


def tac_caption(
    attrs: SpeakerAttrs, script: str, description: str | None = None
) -> str:
    """TAC (single-channel) caption: voice description, blank line, setting
    sentence, blank line, the quoted script.

    ``script`` is the channel's own turns already joined in time order (the
    caller's responsibility, not this function's); it is inserted verbatim
    and is not scanned by ``assert_no_persona`` (see module docstring). A
    literal double quote inside ``script`` passes through unescaped (see
    module docstring).

    ``description``, if given, overrides the template ``voice_description``
    output (e.g. an ``apply_paraphrase_overlay`` result) and is used
    verbatim. It is still checked with ``assert_no_persona`` here, in
    addition to any validation the overlay already performed, as defense in
    depth. Raises ``ValueError`` if ``script`` is empty or whitespace-only.
    """
    if not script.strip():
        raise ValueError("tac_caption: script must not be empty or whitespace-only")
    if description is not None:
        assert_no_persona(description)
        voice_text = description
    else:
        voice_text = voice_description(attrs)
    return (
        f"{voice_text}\n\n" f"{setting_sentence()}\n\n" f'The speaker says: "{script}"'
    )


def _assign_speaker_labels(
    ordered_turns: Sequence[Turn],
) -> tuple[dict[str, str], list[Turn]]:
    """Sort ``ordered_turns`` by start time and assign "Speaker N" labels in
    first-appearance order.

    Returns ``(speaker_id -> label, turns sorted by start time)``. Sorting is
    done here rather than trusted from the caller, so mono_caption/cot_block
    are correct regardless of the order turns were collected in.
    """
    sorted_turns = sorted(ordered_turns, key=lambda t: t.start)
    labels: dict[str, str] = {}
    for turn in sorted_turns:
        if turn.speaker not in labels:
            labels[turn.speaker] = f"Speaker {len(labels) + 1}"
    return labels, sorted_turns


def mono_caption(
    attrs_by_label: dict[str, SpeakerAttrs],
    ordered_turns: Sequence[Turn],
    descriptions: dict[str, str] | None = None,
) -> str:
    """Native multi-talker (mono, single combined stream) caption.

    ``attrs_by_label`` is keyed by the raw speaker id as it appears on
    ``Turn.speaker`` (not by "Speaker N" - those labels are derived here from
    first-appearance order in true temporal order). Layout: one
    "Speaker N: <description>" line per speaker (first-appearance order),
    blank line, then "Speaker N says: "<text>"" lines in true temporal
    (start-time) order, one line per turn. A literal double quote inside a
    turn's text passes through unescaped (see module docstring).

    ``descriptions``, if given, is keyed by the same speaker id/key as
    ``attrs_by_label`` (e.g. an ``apply_paraphrase_overlay`` result) and
    overrides the template ``voice_description`` for that speaker, used
    verbatim (no leading-article stripping, unlike the template path). A
    speaker id absent from ``descriptions`` falls back to the template.
    Every description used (override or template) is checked with
    ``assert_no_persona`` here, in addition to any validation the overlay
    already performed, as defense in depth.

    Raises ``ValueError`` if ``ordered_turns`` is empty, if
    ``attrs_by_label`` is empty, or if any turn's text is empty or
    whitespace-only.
    """
    if not ordered_turns:
        raise ValueError("mono_caption: ordered_turns must not be empty")
    if not attrs_by_label:
        raise ValueError("mono_caption: attrs_by_label must not be empty")
    for turn in ordered_turns:
        if not turn.text.strip():
            raise ValueError(
                f"mono_caption: turn for speaker {turn.speaker!r} has empty "
                "or whitespace-only text"
            )

    labels, sorted_turns = _assign_speaker_labels(ordered_turns)
    descriptions = descriptions or {}

    def _description_for(sid: str) -> str:
        if sid in descriptions:
            override = descriptions[sid]
            assert_no_persona(override)
            return override
        return _strip_leading_article(voice_description(attrs_by_label[sid]))

    voice_lines = "\n".join(
        f"{label}: {_description_for(sid)}" for sid, label in labels.items()
    )
    turn_lines = "\n".join(
        f'{labels[turn.speaker]} says: "{turn.text}"' for turn in sorted_turns
    )
    return f"{voice_lines}\n\n{turn_lines}"


def cot_block(kind: str, **kwargs) -> str:
    """Short ``<think>...</think>`` CoT block, style-matched to the real SFT
    CoT openings (voice/setting planning) but kept to 1-2 sentences per the
    project's short-CoT design decision.

    ``kind == "tac"``: pass ``attrs`` (a ``SpeakerAttrs``). Two-sentence plan
    restating the voice and the single-channel conversational setting.

    ``kind == "mono"``: pass ``attrs_by_label`` and ``ordered_turns`` (same
    shapes as ``mono_caption``). Restates both voices and that turns
    alternate between the speakers.
    """
    if kind == "tac":
        attrs: SpeakerAttrs = kwargs["attrs"]
        plan = (
            f"The speaker is a {_voice_clause(attrs)}. I will continue this "
            "single speaker's side of the conversation in that voice, with "
            "no other speaker in this channel."
        )
    elif kind == "mono":
        attrs_by_label: dict[str, SpeakerAttrs] = kwargs["attrs_by_label"]
        ordered_turns: Sequence[Turn] = kwargs["ordered_turns"]
        labels, _ = _assign_speaker_labels(ordered_turns)
        voice_clause = "; ".join(
            f"{label} is a {_voice_clause(attrs_by_label[speaker_id])}"
            for speaker_id, label in labels.items()
        )
        plan = (
            f"{voice_clause}. The turns alternate between these speakers "
            "in the order given."
        )
    else:
        raise ValueError(f"cot_block: unknown kind {kind!r}, expected 'tac' or 'mono'")

    text = f"<think>\n{plan}\n</think>"
    assert_no_persona(text)
    return text


def apply_paraphrase_overlay(
    captions: dict[str, str], overlay_path: str | Path
) -> dict[str, str]:
    """Replace template voice descriptions with paraphrases from a JSON overlay.

    ``captions`` maps speaker id to its per-speaker voice description ONLY
    (the ``voice_description`` output, not a full assembled tac/mono
    caption). ``overlay_path`` is a JSON file ``{speaker_id: paraphrase}``.
    Speaker ids present in ``captions`` but absent from the overlay keep
    their template description unchanged.

    Each overlay entry is validated and, on failure, raises ``ValueError``
    naming the offending speaker id:
    - the paraphrase must be a non-empty string (after stripping whitespace);
    - it must be single-paragraph (no ``\\n``);
    - it must not contain a persona/character marker (``assert_no_persona``).
    An overlay key not present in ``captions`` also raises ``ValueError``
    naming that speaker, rather than silently creating a new entry (protects
    against a typo'd speaker id in the overlay file).
    """
    overlay = json.loads(Path(overlay_path).read_text())

    unknown = sorted(set(overlay) - set(captions))
    if unknown:
        raise ValueError(
            "apply_paraphrase_overlay: overlay references speaker id(s) not "
            f"present in captions: {unknown!r}"
        )

    result = dict(captions)
    for speaker_id, paraphrase in overlay.items():
        if not isinstance(paraphrase, str) or not paraphrase.strip():
            raise ValueError(
                f"apply_paraphrase_overlay: empty paraphrase for speaker "
                f"{speaker_id!r}"
            )
        if "\n" in paraphrase:
            raise ValueError(
                f"apply_paraphrase_overlay: paraphrase for speaker "
                f"{speaker_id!r} must be a single paragraph (no newlines)"
            )
        try:
            assert_no_persona(paraphrase)
        except ValueError as exc:
            raise ValueError(
                f"apply_paraphrase_overlay: paraphrase for speaker "
                f"{speaker_id!r} contains a persona marker: {exc}"
            ) from exc
        result[speaker_id] = paraphrase

    return result
