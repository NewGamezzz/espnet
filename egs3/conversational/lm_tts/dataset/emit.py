"""Dialogue record emission (Task 5): TAC per-channel records and the mono
combined-stream record, in the exact SFT byte-shape documented in
``docs/bagpiper-findings.md`` ("SFT data schema" section).

Pure assembly module: no I/O. Consumes a windowed ``WindowRecord`` (Task 2),
the corpus-wide per-speaker ``SpeakerAttrs`` (Task 3), the ``WindowAudio``
wav paths already cut for that window (Task 2 audio tail), and renders
captions/CoT via ``dataset.captions`` (Task 4).

``descriptions``, when given, is forwarded verbatim to ``tac_caption`` /
``mono_caption`` as their own ``description``/``descriptions`` override
(pre-formatted, article-free paraphrase text - see those functions'
docstrings). It must NOT be populated from a plain ``voice_description(attrs)``
call: that template output still carries its leading "A "/"An " article,
and routing it back through the override path skips the article-stripping
the template path performs, corrupting the "Speaker N: <description>" line
in ``mono_caption``. Passing ``None`` (the default) always renders the
correct template path from ``attrs``; this module never fabricates a
``descriptions`` dict on its own. "Freezing" a speaker's voice description
(brief requirement) is achieved upstream by measuring ``SpeakerAttrs`` once
per speaker and reusing the same object across every window, not by
threading rendered text through this override.
"""

from __future__ import annotations

from typing import Mapping

from .captions import cot_block, mono_caption, tac_caption
from .preprocessing.attributes import SpeakerAttrs
from .preprocessing.audio import WindowAudio
from .preprocessing.windows import WindowRecord

# The narration regime's proven system prompt (verbatim, matches the real
# dev_multi_talker SFT records - see findings doc).
SYSTEM_PROMPT = "You are a multi-talker text-to-speech system."


def _channel_speakers(window: WindowRecord) -> dict[int, str]:
    """channel -> speaker id, for channels with >= 1 turn.

    Raises ``ValueError`` if a channel carries turns from more than one
    speaker id - channel identity must map to exactly one speaker; anything
    else is a data error upstream (Task 1's manifest parsing / merge_turns).
    """
    mapping: dict[int, str] = {}
    for t in window.turns:
        prev = mapping.get(t.channel)
        if prev is not None and prev != t.speaker:
            raise ValueError(
                f"window {window.window_id!r} channel {t.channel} has turns "
                f"from multiple speakers ({prev!r} and {t.speaker!r}); a "
                "channel must map to exactly one speaker"
            )
        mapping[t.channel] = t.speaker
    return mapping


def is_tac_eligible(window: WindowRecord) -> bool:
    """True iff ``window`` has >= 2 active (turn-bearing) channels."""
    return len(_channel_speakers(window)) >= 2


def emit_tac_records(
    window: WindowRecord,
    attrs_by_speaker: Mapping[str, SpeakerAttrs],
    window_audio: WindowAudio,
    descriptions: dict[str, str] | None = None,
) -> list[dict]:
    """One TAC record per active channel, or ``[]`` if fewer than 2 are active.

    ``script`` for a channel is that channel's own turn texts, joined with
    single spaces, in start-time order (turns are already assigned to this
    window in sorted (start, channel) order by ``build_windows``, but the
    per-channel filter is re-sorted by start defensively).
    """
    channel_speakers = _channel_speakers(window)
    if len(channel_speakers) < 2:
        return []

    records: list[dict] = []
    for channel in sorted(channel_speakers):
        speaker_id = channel_speakers[channel]
        channel_turns = sorted(
            (t for t in window.turns if t.channel == channel), key=lambda t: t.start
        )
        script = " ".join(t.text for t in channel_turns)
        attrs = attrs_by_speaker[speaker_id]
        description = (descriptions or {}).get(speaker_id)
        caption = tac_caption(attrs, script, description=description)
        cot = cot_block("tac", attrs=attrs)
        audio_path = str(window_audio.channel_paths[channel])
        records.append(
            {
                "example_id": f"sssd_tac_{window.window_id}_ch{channel}",
                "messages": [
                    ["system", "text", SYSTEM_PROMPT],
                    ["user", "text", caption],
                    ["assistant", "text", cot],
                    ["assistant", "audio", audio_path],
                ],
                "metadata": {
                    "conv_id": window.window_id,
                    "channel": channel,
                    "num_channels": window.num_channels,
                    "speaker": speaker_id,
                    "t0": window.t0,
                    "t1": window.t1,
                },
            }
        )
    return records


def emit_mono_record(
    window: WindowRecord,
    attrs_by_speaker: Mapping[str, SpeakerAttrs],
    window_audio: WindowAudio,
    descriptions: dict[str, str] | None = None,
) -> dict:
    """The mono (native multi-talker, single combined stream) record.

    Kept for any window with >= 1 active speaker; ``build_windows`` already
    drops empty (zero-turn) windows, so every ``WindowRecord`` reaching this
    function qualifies. ``metadata["speakers"]`` lists speaker ids in
    first-appearance (start-time) order, matching the "Speaker N" label
    assignment ``mono_caption``/``cot_block`` derive internally.
    """
    channel_speakers = _channel_speakers(window)
    turns = list(window.turns)
    attrs_by_label = {sid: attrs_by_speaker[sid] for sid in channel_speakers.values()}

    caption = mono_caption(attrs_by_label, turns, descriptions=descriptions)
    cot = cot_block("mono", attrs_by_label=attrs_by_label, ordered_turns=turns)
    audio_path = str(window_audio.mix_path)

    ordered_speakers: list[str] = []
    for t in sorted(turns, key=lambda t: t.start):
        if t.speaker not in ordered_speakers:
            ordered_speakers.append(t.speaker)

    return {
        "example_id": f"sssd_mono_{window.window_id}",
        "messages": [
            ["system", "text", SYSTEM_PROMPT],
            ["user", "text", caption],
            ["assistant", "text", cot],
            ["assistant", "audio", audio_path],
        ],
        "metadata": {
            "conv_id": window.window_id,
            "variant": "mono",
            "speakers": ordered_speakers,
            "t0": window.t0,
            "t1": window.t1,
        },
    }
