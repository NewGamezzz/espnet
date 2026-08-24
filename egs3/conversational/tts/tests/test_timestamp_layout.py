import pytest

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    FRAMES_PER_SECOND, build_branch_texts_timestamped, turn_frame_spans,
)
from egs3.conversational.tts.src.timestamp_layout import (
    TimestampLayout, prompt_window_layout, synthesize_layout,
)

FPS = FRAMES_PER_SECOND


def _t(ch, text, i):
    return Turn(ch, f"spk{ch}", text, float(i), float(i))  # ordinal, like CoVoMix2


class TestSynthesizeLayout:
    def test_sequential_gap_separated_frames(self):
        turns = [_t(0, "abc def", 0), _t(1, "bead cab", 1), _t(0, "chad", 2)]
        lay = synthesize_layout(turns, [2.0, 3.0, 1.0], gap_sec=0.4)
        assert lay.gap_frames == round(0.4 * FPS) == 38
        assert lay.turn_frames == [round(2.0 * FPS), round(3.0 * FPS), round(1.0 * FPS)]
        spans = turn_frame_spans(lay.turns, 0.0, 10_000)
        assert spans[0] == (0, 188)
        assert spans[1] == (188 + 38, 188 + 38 + 281)
        assert spans[2][0] == spans[1][1] + 38
        assert all(t.channel == u.channel and t.text == u.text for t, u in zip(lay.turns, turns))

    def test_chunk_spans_tile_the_timeline(self):
        turns = [_t(0, "a", 0), _t(1, "b", 1), _t(0, "c", 2), _t(1, "d", 3)]
        lay = synthesize_layout(turns, [1.0, 1.5, 1.0, 2.0], gap_sec=0.4)
        s0, n0 = lay.chunk_span(0, 2)
        s1, n1 = lay.chunk_span(2, 4)
        assert s0 == 0 and s0 + n0 == s1
        assert n0 == lay.turn_frames[0] + lay.turn_frames[1] + 2 * lay.gap_frames
        assert s1 + n1 == sum(lay.turn_frames) + 4 * lay.gap_frames

    def test_layout_builds_mode_t_text_for_a_chunk(self):
        turns = [_t(0, "abc", 0), _t(1, "de", 1)]
        lay = synthesize_layout(turns, [1.0, 1.0], gap_sec=0.4)
        s, n = lay.chunk_span(0, 2)
        out = build_branch_texts_timestamped(lay.turns, 2, t0=s / FPS, target_frames=n)
        assert len(out[0]) == len(out[1]) == n
        assert out[0][0] == "<turn>" and out[1][lay.turn_frames[0] + lay.gap_frames] == "<turn>"

    def test_zero_gap_allowed(self):
        lay = synthesize_layout([_t(0, "a", 0), _t(1, "b", 1)], [1.0, 1.0], gap_sec=0.0)
        assert lay.gap_frames == 0 and lay.chunk_span(0, 2)[1] == 2 * round(1.0 * FPS)

    def test_unfittable_turn_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            synthesize_layout([_t(0, "way too many characters", 0)], [0.05], gap_sec=0.4)

    def test_negative_gap_rejected(self):
        with pytest.raises(ValueError, match="gap_sec"):
            synthesize_layout([_t(0, "a", 0)], [1.0], gap_sec=-0.1)


class TestPromptWindowLayout:
    def test_prompt_blocks_then_shifted_window_turns(self):
        prompt = [Turn(0, "a", "cage jade", 25.5, 28.0), Turn(1, "b", "badge fig", 28.5, 31.0)]
        window = [Turn(0, "a", "abc def", 5.5, 8.0), Turn(1, "b", "bead cab", 8.5, 11.0)]
        fs = 24000
        blocks = [60000, 60000]  # 2.5 s each
        out = prompt_window_layout(prompt, blocks, window, window_t0=5.0, fs=fs)
        assert [t.text for t in out] == ["cage jade", "badge fig", "abc def", "bead cab"]
        assert (out[0].start, out[0].end) == (0.0, 2.5)
        assert (out[1].start, out[1].end) == (2.5, 5.0)
        assert (out[2].start, out[2].end) == (5.0 + 0.5, 5.0 + 3.0)
        assert (out[3].start, out[3].end) == (5.0 + 3.5, 5.0 + 6.0)

    def test_window_turn_starting_before_t0_is_clamped_to_prompt_end(self):
        prompt = [Turn(0, "a", "x", 25.0, 26.0)]
        window = [Turn(0, "a", "late", 4.0, 7.0)]
        out = prompt_window_layout(prompt, [24000], window, window_t0=5.0, fs=24000)
        assert out[1].start == 1.0 and out[1].end == 3.0
