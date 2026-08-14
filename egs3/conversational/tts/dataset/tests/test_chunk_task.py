import random

import pytest

from egs3.conversational.tts.dataset.preprocessing.chunk_task import (
    ChunkTaskParams,
    ChunkTaskPlan,
    draw_chunk_task,
    plan_from_json,
    plan_to_json,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    from_json,
    to_json,
)


def _turns():
    return (
        Turn(channel=0, speaker="A", text="hi there", start=100.0, end=102.0),
        Turn(channel=1, speaker="B", text="hello", start=102.0, end=104.0),
    )


def _record(chunk_task=None):
    return WindowRecord(
        window_id="s1_w00003",
        session_id="s1",
        audio_relpath="a.flac",
        num_channels=2,
        sample_rate=48000,
        t0=100.0,
        t1=130.0,
        turns=_turns(),
        chunk_task=chunk_task,
    )


def test_plan_json_roundtrip():
    plan = ChunkTaskPlan(
        kind="full", prev_span=(94.0, 100.0), prompt_spans=((10.0, 14.5), (40.0, 44.5))
    )
    assert plan_from_json(plan_to_json(plan)) == plan


def test_prompt_only_has_no_prev_span():
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="prompt_only",
            prev_span=(94.0, 100.0),
            prompt_spans=((10.0, 14.5), (40.0, 44.5)),
        )
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="full", prev_span=None, prompt_spans=((10.0, 14.5), (40.0, 44.5))
        )


def test_prompt_spans_equal_length_enforced():
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="prompt_only",
            prev_span=None,
            prompt_spans=((10.0, 14.5), (40.0, 43.0)),
        )


def test_window_record_json_omits_absent_chunk_task():
    d = to_json(_record())
    assert "chunk_task" not in d
    assert from_json(d).chunk_task is None


def test_window_record_json_roundtrips_chunk_task():
    plan = ChunkTaskPlan(
        kind="prompt_only", prev_span=None, prompt_spans=((10.0, 14.5), (40.0, 44.5))
    )
    rec = _record(chunk_task=plan)
    d = to_json(rec)
    assert d["chunk_task"]["kind"] == "prompt_only"
    assert from_json(d).chunk_task == plan


def test_draw_full_task_spans_are_disjoint_and_floored():
    # ch0 speaks at 10-16s and 200-206s; ch1 at 20-26s and 210-216s; window 100-130
    turns = (
        Turn(0, "A", "aaaa", 10.0, 16.0),
        Turn(1, "B", "bbbb", 20.0, 26.0),
        Turn(0, "A", "cccc", 200.0, 206.0),
        Turn(1, "B", "dddd", 210.0, 216.0),
    )
    rec = _record()
    plan = draw_chunk_task(
        rec,
        turns,
        300.0,
        (),
        2,
        random.Random(1),
        ChunkTaskParams(prompt_only_prob=0.0),
    )
    assert plan.kind == "full"
    a, b = plan.prev_span
    assert b == rec.t0 and 2.0 <= b - a <= 10.0
    lengths = {round(e - s, 6) for s, e in plan.prompt_spans}
    assert len(lengths) == 1 and 3.0 <= lengths.pop() <= 8.0
    forbidden = (a, rec.t1)
    for s, e in plan.prompt_spans:
        assert e <= forbidden[0] or s >= forbidden[1]


def test_speech_floor_binds_on_speech_not_slice():
    # ch1's only outside turn is 2.5 s long: cannot reach 3.0 s speech -> None
    turns = (Turn(0, "A", "aaaa", 10.0, 16.0), Turn(1, "B", "bb", 20.0, 22.5))
    assert (
        draw_chunk_task(
            _record(),
            turns,
            300.0,
            (),
            2,
            random.Random(1),
            ChunkTaskParams(prompt_only_prob=0.0),
        )
        is None
    )


def test_h_clamped_by_session_start_degrades_to_prompt_only():
    rec = WindowRecord(
        window_id="w",
        session_id="s",
        audio_relpath="a.flac",
        num_channels=2,
        sample_rate=48000,
        t0=1.0,
        t1=25.0,
        turns=_turns(),
    )
    turns = (Turn(0, "A", "aaaa", 40.0, 46.0), Turn(1, "B", "bbbb", 50.0, 56.0))
    plan = draw_chunk_task(
        rec,
        turns,
        300.0,
        (),
        2,
        random.Random(1),
        ChunkTaskParams(prompt_only_prob=0.0),
    )
    assert plan is not None and plan.kind == "prompt_only" and plan.prev_span is None


def test_determinism():
    turns = (Turn(0, "A", "aaaa", 10.0, 16.0), Turn(1, "B", "bbbb", 20.0, 26.0))
    args = (_record(), turns, 300.0, (), 2)
    p1 = draw_chunk_task(*args, random.Random(7), ChunkTaskParams())
    p2 = draw_chunk_task(*args, random.Random(7), ChunkTaskParams())
    assert p1 == p2


def test_exclusion_span_clamps_prev_start_to_its_end():
    # With seed 1 and prompt_only_prob=0.0, the drawn prev-slice length puts
    # the raw prev_start at ~91.22 (t0=100). An exclusion span (93, 95)
    # overlaps (prev_start, t0), so prev_start must clamp up to 95.0 - well
    # short of the 0.0 session-start clamp, isolating this clamp path.
    turns = (
        Turn(0, "A", "aaaa", 10.0, 16.0),
        Turn(1, "B", "bbbb", 20.0, 26.0),
        Turn(0, "A", "cccc", 200.0, 206.0),
        Turn(1, "B", "dddd", 210.0, 216.0),
    )
    rec = _record()
    plan = draw_chunk_task(
        rec,
        turns,
        300.0,
        ((93.0, 95.0),),
        2,
        random.Random(1),
        ChunkTaskParams(prompt_only_prob=0.0),
    )
    assert plan.kind == "full"
    assert plan.prev_span == (95.0, 100.0)


def test_prompt_anchor_truncated_by_forbidden_region():
    # Single channel, forced prompt_only (F = (t0, t1) = (100, 130)). The one
    # candidate anchor at 95-99 has 4 s of speech (floor 3.0 reached at 95+3)
    # but only 5 s of headroom before F starts at 100, so A_c=5.0 caps the
    # drawn prompt length (~7.24 s for seed 1) well below prompt_slice_max.
    turns = (Turn(0, "A", "aaaa", 95.0, 99.0),)
    rec = _record()
    plan = draw_chunk_task(
        rec,
        turns,
        300.0,
        (),
        1,
        random.Random(1),
        ChunkTaskParams(prompt_only_prob=1.0),
    )
    assert plan.kind == "prompt_only"
    assert plan.prompt_spans == ((95.0, 100.0),)


def test_cross_channel_lp_bound_by_one_channels_floor_capped_by_anothers_headroom():
    # Deliberately asymmetric 2-channel geometry (forced prompt_only, so
    # F = (t0, t1) = (100, 130)); hand-verified against the real rng draw
    # sequence for seed 15 (not just the final assertion):
    #   ch0: single candidate turn 95-99 (4 s) -> m_c=3.0 (floor reached at
    #     95+3), a_c=min(prompt_slice_max=25, 100-95)=5.0 (TIGHT headroom).
    #   ch1: two candidates, turn 10-11 (1 s) then turn 12-16 (4 s).
    #     rng.sample's seed-15 order tries anchor=10 first: walking forward,
    #     the 10-11 turn supplies 1 s of speech, the 12-16 turn supplies the
    #     remaining 2 s at offset (12-10)+2=4, so m_c=4.0 (<= its own
    #     a_c=min(25, 100-10)=25.0, so this anchor is individually valid and
    #     is kept - the other candidate, anchor=12 with m_c=3.0, is never
    #     reached).
    # Cross-channel reduction: min_a = min(5.0, 25.0) = 5.0 (ch0's headroom
    #   is the binding cap); max_m = max(3.0, 4.0) = 4.0 (ch1's floor is the
    #   binding minimum) - the two bounds come from DIFFERENT channels, so a
    #   min/max mix-up (e.g. taking max(a_values) or min(m_values), or
    #   applying the reduction in the wrong order) would not reproduce this
    #   result by coincidence the way a same-channel-bound case could.
    # lp_draw for seed 15 (uniform(1, 25)) is ~1.28, below both bounds:
    #   Lp = min(1.28, min_a=5.0) = 1.28, then Lp = max(1.28, max_m=4.0) =
    #   4.0. Since 4.0 <= min_a(5.0), the plan is valid with Lp = 4.0 on
    #   both channels (not equal to lp_draw, not equal to either channel's
    #   own a_c - genuinely bound by the OTHER channel's floor).
    turns = (
        Turn(0, "A", "x", 95.0, 99.0),
        Turn(1, "B", "y", 10.0, 11.0),
        Turn(1, "B", "z", 12.0, 16.0),
    )
    params = ChunkTaskParams(
        prompt_only_prob=1.0,
        prompt_slice_min=1.0,
        prompt_slice_max=25.0,
        prompt_speech_floor=3.0,
    )
    plan = draw_chunk_task(_record(), turns, 300.0, (), 2, random.Random(15), params)
    assert plan.kind == "prompt_only"
    assert plan.prompt_spans == ((95.0, 99.0), (10.0, 14.0))


def test_cross_channel_floor_conflict_returns_none():
    # Same ch0 as above (m_c=3.0, a_c=5.0, hand-verified there). ch1's floor
    # requirement is now pushed past ch0's headroom: turn 10-10.5 (0.5 s)
    # then turn 13-17 (4 s). Seed 0's rng.sample order for ch1 also tries
    # anchor=10 first: 0.5 s of speech from the first turn, the remaining
    # 2.5 s reached at offset (13-10)+2.5=5.5, so m_c=5.5 - still <= this
    # anchor's own a_c=min(25, 100-10)=25.0, so it is individually valid and
    # kept (never falls through to the anchor=13 candidate).
    # min_a = min(5.0, 25.0) = 5.0; max_m = max(3.0, 5.5) = 5.5. lp_draw for
    #   seed 0 (uniform(1, 25)) is ~19.19: Lp = min(19.19, 5.0) = 5.0, then
    #   Lp = max(5.0, 5.5) = 5.5, which EXCEEDS min_a(5.0) - ch1's floor
    #   genuinely conflicts with ch0's headroom (no single length satisfies
    #   both channels at once), so the draw must return None rather than
    #   silently emitting a plan clamped to 5.0 (which a min/max swap bug
    #   would do, since min(max(lp_draw, max_m), min_a) = 5.0 here).
    turns = (
        Turn(0, "A", "x", 95.0, 99.0),
        Turn(1, "B", "y", 10.0, 10.5),
        Turn(1, "B", "z", 13.0, 17.0),
    )
    params = ChunkTaskParams(
        prompt_only_prob=1.0,
        prompt_slice_min=1.0,
        prompt_slice_max=25.0,
        prompt_speech_floor=3.0,
    )
    assert (
        draw_chunk_task(_record(), turns, 300.0, (), 2, random.Random(0), params)
        is None
    )
