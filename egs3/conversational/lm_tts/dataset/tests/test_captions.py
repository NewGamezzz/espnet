"""Tests for narration-style caption/CoT templates and the persona guard.

Golden-string style: expected outputs are written out in full so any wording
drift is a conscious test edit, not a silent regression. See
``dataset/captions.py`` for the design rationale (decision 14: no
persona/character framing).
"""

import json

import pytest

from dataset.captions import (
    PERSONA_MARKERS,
    PITCH_ADJ,
    RATE_PHRASE,
    VARIABILITY_ADJ,
    apply_paraphrase_overlay,
    assert_no_persona,
    cot_block,
    mono_caption,
    setting_sentence,
    tac_caption,
    voice_description,
)
from dataset.preprocessing.attributes import SpeakerAttrs
from dataset.preprocessing.sssd import Turn


def attrs(gender, pitch_band, variability_band, rate_band):
    """Build a SpeakerAttrs with arbitrary (irrelevant to captions) raw numbers."""
    return SpeakerAttrs(
        median_f0=180.0,
        f0_iqr=20.0,
        words_per_sec=3.0,
        pitch_band=pitch_band,
        variability_band=variability_band,
        rate_band=rate_band,
        gender=gender,
        gender_source="metadata",
    )


ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK = attrs("female", "high", "expressive", "brisk")
ATTRS_MALE_LOW_FLAT_MEASURED = attrs("male", "low", "flat", "measured")
ATTRS_FEMALE_MEDIUM_FLAT_MODERATE = attrs("female", "medium", "flat", "moderate")


class TestBandAdjectiveTable:
    """The band -> vocabulary mapping is fixed, deterministic, module-level."""

    def test_pitch_adjectives(self):
        assert PITCH_ADJ == {"low": "deep", "medium": "medium", "high": "high"}

    def test_variability_adjectives(self):
        assert VARIABILITY_ADJ == {
            "flat": "calm, even",
            "expressive": "lively, expressive",
        }

    def test_rate_phrases(self):
        assert RATE_PHRASE == {
            "measured": "an unhurried pace",
            "moderate": "a natural pace",
            "brisk": "a quick pace",
        }


class TestVoiceDescription:
    """Golden strings for each band combination exercised."""

    def test_female_high_expressive_brisk(self):
        text = voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)
        assert text == (
            "A female speaker with a high-pitched, lively, expressive voice, "
            "speaking at a quick pace in a clean close-microphone recording."
        )
        assert_no_persona(text)

    def test_male_low_flat_measured(self):
        text = voice_description(ATTRS_MALE_LOW_FLAT_MEASURED)
        assert text == (
            "A male speaker with a deep-pitched, calm, even voice, "
            "speaking at an unhurried pace in a clean close-microphone recording."
        )
        assert_no_persona(text)

    def test_female_medium_flat_moderate(self):
        text = voice_description(ATTRS_FEMALE_MEDIUM_FLAT_MODERATE)
        assert text == (
            "A female speaker with a medium-pitched, calm, even voice, "
            "speaking at a natural pace in a clean close-microphone recording."
        )
        assert_no_persona(text)

    def test_no_em_dash_in_output(self):
        for a in (
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            ATTRS_MALE_LOW_FLAT_MEASURED,
            ATTRS_FEMALE_MEDIUM_FLAT_MODERATE,
        ):
            assert "—" not in voice_description(a)


class TestSettingSentence:
    def test_fixed_zero_partner_content(self):
        text = setting_sentence()
        assert text == (
            "The following is one side of a natural two-person conversation; "
            "the speaker responds in their own turns."
        )
        assert_no_persona(text)


class TestTacCaption:
    def test_golden(self):
        caption = tac_caption(
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK, "We're in Istanbul, Turkey."
        )
        assert caption == (
            "A female speaker with a high-pitched, lively, expressive voice, "
            "speaking at a quick pace in a clean close-microphone recording.\n\n"
            "The following is one side of a natural two-person conversation; "
            "the speaker responds in their own turns.\n\n"
            'The speaker says: "We\'re in Istanbul, Turkey."'
        )
        assert_no_persona(caption)

    def test_uses_straight_quotes_only(self):
        caption = tac_caption(ATTRS_MALE_LOW_FLAT_MEASURED, "Hello there.")
        assert "“" not in caption
        assert "”" not in caption
        assert '"' in caption

    def test_no_em_dash(self):
        caption = tac_caption(ATTRS_MALE_LOW_FLAT_MEASURED, "Hello there.")
        assert "—" not in caption


class TestMonoCaption:
    """Speaker labels assigned in first-appearance order of true temporal order,
    not the order turns happen to be passed in."""

    def _turns_scrambled(self):
        turn_first = Turn(
            channel=0,
            speaker="spk_f",
            text="We're in Istanbul, Turkey.",
            start=0.0,
            end=2.0,
        )
        turn_second = Turn(
            channel=1,
            speaker="spk_m",
            text="The food culture here will blow your mind.",
            start=2.0,
            end=5.0,
        )
        turn_third = Turn(
            channel=0,
            speaker="spk_f",
            text="Let's get food hunting.",
            start=5.0,
            end=6.5,
        )
        # Deliberately out of temporal order to prove mono_caption sorts by
        # start time itself rather than trusting input order.
        return [turn_third, turn_first, turn_second]

    def test_golden(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        caption = mono_caption(attrs_by_label, self._turns_scrambled())
        assert caption == (
            "Speaker 1: female speaker with a high-pitched, lively, expressive "
            "voice, speaking at a quick pace in a clean close-microphone "
            "recording.\n"
            "Speaker 2: male speaker with a deep-pitched, calm, even voice, "
            "speaking at an unhurried pace in a clean close-microphone "
            "recording.\n\n"
            'Speaker 1 says: "We\'re in Istanbul, Turkey."\n'
            'Speaker 2 says: "The food culture here will blow your mind."\n'
            'Speaker 1 says: "Let\'s get food hunting."'
        )
        assert_no_persona(caption)

    def test_no_em_dash(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        caption = mono_caption(attrs_by_label, self._turns_scrambled())
        assert "—" not in caption


class TestCotBlock:
    def test_tac_golden(self):
        block = cot_block("tac", attrs=ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)
        assert block == (
            "<think>\n"
            "The speaker is a female speaker with a high-pitched, lively, "
            "expressive voice, speaking at a quick pace in a clean "
            "close-microphone recording. I will continue this single "
            "speaker's side of the conversation in that voice, with no "
            "other speaker in this channel.\n"
            "</think>"
        )
        assert_no_persona(block)

    def test_mono_golden(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        turns = [
            Turn(channel=0, speaker="spk_f", text="Hi.", start=0.0, end=1.0),
            Turn(channel=1, speaker="spk_m", text="Hey.", start=1.0, end=2.0),
        ]
        block = cot_block("mono", attrs_by_label=attrs_by_label, ordered_turns=turns)
        assert block == (
            "<think>\n"
            "Speaker 1 is a female speaker with a high-pitched, lively, "
            "expressive voice, speaking at a quick pace in a clean "
            "close-microphone recording; Speaker 2 is a male speaker with a "
            "deep-pitched, calm, even voice, speaking at an unhurried pace "
            "in a clean close-microphone recording. The turns alternate "
            "between these speakers in the order given.\n"
            "</think>"
        )
        assert_no_persona(block)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            cot_block("duet", attrs=ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)


class TestPersonaGuard:
    """Regression guard for decision 14: reject role/character/persona framing."""

    @pytest.mark.parametrize(
        "text",
        [
            "You are role-playing a wizard giving advice.",
            "Embody the character of a battle-scarred veteran.",
            "Adopt the persona of a 1920s radio host.",
            "Pretend to be a nervous student.",
            "Portray a weary detective recounting the case.",
            "ROLE: mentor. Speak with authority.",
        ],
    )
    def test_persona_markers_raise(self, text):
        with pytest.raises(ValueError):
            assert_no_persona(text)

    def test_clean_text_does_not_raise(self):
        assert_no_persona(
            "A female speaker with a high-pitched voice in a clean recording."
        )

    def test_personality_is_not_a_false_positive(self):
        # "persona" is a substring of "personality" but word-boundary anchored
        # matching must not flag it.
        assert_no_persona("Her personality comes through in a lively voice.")

    def test_characteristic_is_not_a_false_positive(self):
        assert_no_persona("A characteristic warmth colors her delivery.")

    def test_persona_markers_pattern_is_case_insensitive(self):
        assert PERSONA_MARKERS.search("She will EMBODY the mentor role.")

    def test_generated_templates_never_contain_persona_markers(self):
        for a in (
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            ATTRS_MALE_LOW_FLAT_MEASURED,
            ATTRS_FEMALE_MEDIUM_FLAT_MODERATE,
        ):
            assert_no_persona(voice_description(a))
            assert_no_persona(cot_block("tac", attrs=a))


class TestTacCaptionInputGuards:
    """Fix 1: degenerate input must raise loudly, not silently produce
    near-empty training data."""

    def test_empty_script_raises(self):
        with pytest.raises(ValueError):
            tac_caption(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK, "")

    def test_whitespace_only_script_raises(self):
        with pytest.raises(ValueError):
            tac_caption(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK, "   \n\t  ")


class TestMonoCaptionInputGuards:
    """Fix 1: degenerate input must raise loudly, not silently produce
    near-empty training data."""

    def test_empty_turns_raises(self):
        with pytest.raises(ValueError):
            mono_caption({"spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK}, [])

    def test_empty_attrs_by_label_raises(self):
        turns = [Turn(channel=0, speaker="spk_f", text="Hi.", start=0.0, end=1.0)]
        with pytest.raises(ValueError):
            mono_caption({}, turns)

    def test_empty_turn_text_raises(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        turns = [
            Turn(channel=0, speaker="spk_f", text="Hi.", start=0.0, end=1.0),
            Turn(channel=1, speaker="spk_m", text="   ", start=1.0, end=2.0),
        ]
        with pytest.raises(ValueError):
            mono_caption(attrs_by_label, turns)


class TestTacCaptionDescriptionOverride:
    """Fix 2: apply_paraphrase_overlay output must be able to reach the
    assembled caption, not just live in a disconnected dict."""

    def test_override_used_verbatim(self):
        caption = tac_caption(
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "We're in Istanbul, Turkey.",
            description="A bright, energetic young woman's voice.",
        )
        assert caption == (
            "A bright, energetic young woman's voice.\n\n"
            "The following is one side of a natural two-person conversation; "
            "the speaker responds in their own turns.\n\n"
            'The speaker says: "We\'re in Istanbul, Turkey."'
        )

    def test_no_override_falls_back_to_template(self):
        caption = tac_caption(
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK, "We're in Istanbul, Turkey."
        )
        assert caption.startswith(voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK))

    def test_persona_override_raises_at_assembly(self):
        with pytest.raises(ValueError):
            tac_caption(
                ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
                "We're in Istanbul, Turkey.",
                description="Embody the character of a queen.",
            )


class TestMonoCaptionDescriptionOverride:
    """Fix 2: same override wiring for the multi-speaker template, keyed by
    the same speaker label/key as attrs_by_label."""

    def _turns(self):
        return [
            Turn(
                channel=0,
                speaker="spk_f",
                text="We're in Istanbul, Turkey.",
                start=0.0,
                end=2.0,
            ),
            Turn(
                channel=1,
                speaker="spk_m",
                text="The food culture here will blow your mind.",
                start=2.0,
                end=5.0,
            ),
        ]

    def test_override_used_verbatim_missing_key_falls_back(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        descriptions = {"spk_f": "A bright, energetic young woman's voice."}
        caption = mono_caption(attrs_by_label, self._turns(), descriptions=descriptions)
        assert caption == (
            "Speaker 1: A bright, energetic young woman's voice.\n"
            "Speaker 2: male speaker with a deep-pitched, calm, even voice, "
            "speaking at an unhurried pace in a clean close-microphone "
            "recording.\n\n"
            'Speaker 1 says: "We\'re in Istanbul, Turkey."\n'
            'Speaker 2 says: "The food culture here will blow your mind."'
        )

    def test_persona_override_raises_at_assembly(self):
        attrs_by_label = {
            "spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            "spk_m": ATTRS_MALE_LOW_FLAT_MEASURED,
        }
        descriptions = {"spk_f": "Embody the character of a queen."}
        with pytest.raises(ValueError):
            mono_caption(attrs_by_label, self._turns(), descriptions=descriptions)


class TestQuotePassthrough:
    """Fix 3: literal double quotes inside verbatim speech pass through
    unescaped. This documents the chosen behavior, not a bug."""

    def test_tac_caption_literal_quotes_pass_through_unescaped(self):
        caption = tac_caption(ATTRS_MALE_LOW_FLAT_MEASURED, 'She said "hello" to me.')
        assert 'The speaker says: "She said "hello" to me."' in caption

    def test_mono_caption_literal_quotes_pass_through_unescaped(self):
        attrs_by_label = {"spk_f": ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK}
        turns = [
            Turn(
                channel=0,
                speaker="spk_f",
                text='She said "hello" to me.',
                start=0.0,
                end=2.0,
            )
        ]
        caption = mono_caption(attrs_by_label, turns)
        assert 'Speaker 1 says: "She said "hello" to me."' in caption


class TestPersonaMarkersInflections:
    """Fix 4: PERSONA_MARKERS must catch common inflections while staying
    word-boundary anchored."""

    @pytest.mark.parametrize(
        "text",
        [
            "She has multiple roles in this play.",
            "The characters in this scene argue.",
            "Multiple personas emerge in the recording.",
            "She embodies the mentor archetype.",
            "He embodied the veteran perfectly.",
            "The actor is embodying a new role.",
            "That is quite the embodiment of grit.",
            "She portrays a weary detective.",
            "He portrayed a nervous student.",
            "The actor is portraying a queen.",
            "The portrayal was convincing.",
            "She pretends to be calm.",
            "He pretended not to notice.",
            "They are pretending everything is fine.",
            "The kids love to roleplay adventures.",
            "The kids love to role-play adventures.",
            "She is roleplaying a detective.",
            "They role-played the whole scenario.",
        ],
    )
    def test_inflections_are_matched(self, text):
        with pytest.raises(ValueError):
            assert_no_persona(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Her personality comes through in a lively voice.",
            "A characteristic warmth colors her delivery.",
            "The story was told in a characterful way.",
            "The rug was unrolled across the floor.",
            "His tone felt impersonal and flat.",
        ],
    )
    def test_no_false_positives_on_lookalike_words(self, text):
        assert_no_persona(text)

    def test_no_false_positives_on_template_vocabulary(self):
        for a in (
            ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK,
            ATTRS_MALE_LOW_FLAT_MEASURED,
            ATTRS_FEMALE_MEDIUM_FLAT_MODERATE,
        ):
            assert_no_persona(voice_description(a))
            assert_no_persona(cot_block("tac", attrs=a))
        assert_no_persona(setting_sentence())


class TestApplyParaphraseOverlay:
    def test_overlay_replaces_present_speaker_keeps_missing_as_template(self, tmp_path):
        captions = {
            "spk_f": voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK),
            "spk_m": voice_description(ATTRS_MALE_LOW_FLAT_MEASURED),
        }
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(
            json.dumps({"spk_f": "A bright, energetic young woman's voice."})
        )
        result = apply_paraphrase_overlay(captions, overlay_path)
        assert result["spk_f"] == "A bright, energetic young woman's voice."
        assert result["spk_m"] == captions["spk_m"]

    def test_empty_paraphrase_raises_naming_speaker(self, tmp_path):
        captions = {"spk_f": voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)}
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(json.dumps({"spk_f": "   "}))
        with pytest.raises(ValueError, match="spk_f"):
            apply_paraphrase_overlay(captions, overlay_path)

    def test_multiline_paraphrase_raises_naming_speaker(self, tmp_path):
        captions = {"spk_f": voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)}
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(json.dumps({"spk_f": "Line one.\nLine two."}))
        with pytest.raises(ValueError, match="spk_f"):
            apply_paraphrase_overlay(captions, overlay_path)

    def test_persona_paraphrase_raises_naming_speaker(self, tmp_path):
        captions = {"spk_f": voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)}
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(
            json.dumps({"spk_f": "Embody the character of a queen."})
        )
        with pytest.raises(ValueError, match="spk_f"):
            apply_paraphrase_overlay(captions, overlay_path)

    def test_unknown_overlay_speaker_raises(self, tmp_path):
        captions = {"spk_f": voice_description(ATTRS_FEMALE_HIGH_EXPRESSIVE_BRISK)}
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(json.dumps({"spk_ghost": "A calm voice."}))
        with pytest.raises(ValueError, match="spk_ghost"):
            apply_paraphrase_overlay(captions, overlay_path)
