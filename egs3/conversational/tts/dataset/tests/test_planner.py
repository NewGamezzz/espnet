import random
from types import SimpleNamespace

from ..preprocessing.chunk_task import ChunkTaskParams
from ..preprocessing.planner import (
    WindowParams,
    _apply_mask_coin,
    plan_session,
    plan_sessions,
)
from ..preprocessing.sessions import SessionRecord
from ..preprocessing.sssd import Recording, Turn
from ..preprocessing.text import timestamp_fits
from ..preprocessing.windows import WindowRecord, WindowingStats, build_windows, to_json

# Long two-channel session with clean alternation and real gaps so several
# windows fit (window params shrunk so the 60 s session yields multiple).
PARAMS = WindowParams(window_min=5.0, window_max=15.0, tail_min=2.0)

# Golden pin for TestChunkTaskPlanning.test_epoch_mode_parity_branch_pinned;
# see that test for how it was captured (a real plan_session(..., epoch=5,
# chunk_params=None) run at this commit, hand-copied from its to_json()
# output - the same pattern test_parity.py uses for its golden JSONL files).
_EPOCH_MODE_GOLDEN = [
    {
        "window_id": "sess_golden_w00000",
        "session_id": "sess_golden",
        "audio_relpath": "original/a.flac",
        "num_channels": 2,
        "sample_rate": 48000,
        "t0": 0.0,
        "t1": 3.5,
        "duration": 3.5,
        "num_active_speakers": 2,
        "channel_speech_sec": [1.5, 1.5],
        "exchange_count": 1,
        "turns": [
            {
                "channel": 0,
                "speaker": "s0",
                "text": "hello there",
                "start": 0.0,
                "end": 1.5,
            },
            {
                "channel": 1,
                "speaker": "s1",
                "text": "hello there",
                "start": 2.0,
                "end": 3.5,
            },
        ],
    },
    {
        "window_id": "sess_golden_w00001",
        "session_id": "sess_golden",
        "audio_relpath": "original/a.flac",
        "num_channels": 2,
        "sample_rate": 48000,
        "t0": 4.0,
        "t1": 7.5,
        "duration": 3.5,
        "num_active_speakers": 2,
        "channel_speech_sec": [1.5, 1.5],
        "exchange_count": 1,
        "turns": [
            {
                "channel": 0,
                "speaker": "s0",
                "text": "hello there",
                "start": 4.0,
                "end": 5.5,
            },
            {
                "channel": 1,
                "speaker": "s1",
                "text": "hello there",
                "start": 6.0,
                "end": 7.5,
            },
        ],
    },
    {
        "window_id": "sess_golden_w00002",
        "session_id": "sess_golden",
        "audio_relpath": "original/a.flac",
        "num_channels": 2,
        "sample_rate": 48000,
        "t0": 8.0,
        "t1": 11.5,
        "duration": 3.5,
        "num_active_speakers": 2,
        "channel_speech_sec": [1.5, 1.5],
        "exchange_count": 1,
        "turns": [
            {
                "channel": 0,
                "speaker": "s0",
                "text": "hello there",
                "start": 8.0,
                "end": 9.5,
            },
            {
                "channel": 1,
                "speaker": "s1",
                "text": "hello there",
                "start": 10.0,
                "end": 11.5,
            },
        ],
    },
]


def _turns(n=30, step=2.0):
    out = []
    for i in range(n):
        start = i * step
        out.append(
            Turn(
                channel=i % 2,
                speaker=f"s{i % 2}",
                text="hello there",
                start=start,
                end=start + step - 0.5,
            )
        )
    return tuple(out)


def _session(**kw):
    turns = kw.pop("turns", _turns())
    duration = kw.pop("duration", max(t.end for t in turns) + 0.5)
    base = dict(
        session_id="sess_a",
        audio_relpath="original/a.flac",
        num_channels=2,
        sample_rate=48000,
        duration=duration,
        turns=turns,
    )
    base.update(kw)
    return SessionRecord(**base)


class TestFrozenMode:
    def test_matches_build_windows_with_legacy_seed(self):
        s = _session()
        planned, _ = plan_session(s, params=PARAMS, seed=0, epoch=None)
        rec = Recording(
            id=s.session_id,
            audio_relpath=s.audio_relpath,
            sample_rate=s.sample_rate,
            num_channels=s.num_channels,
            duration=s.duration,
        )
        direct, _ = build_windows(
            s.session_id,
            rec,
            s.turns,
            window_min=PARAMS.window_min,
            window_max=PARAMS.window_max,
            boundary_guard=PARAMS.boundary_guard,
            tail_min=PARAMS.tail_min,
            rng=random.Random("0:window:sess_a"),
            trim_to_turns=PARAMS.trim_to_turns,
            min_coverage=PARAMS.min_coverage,
            snap_start_to_turn=PARAMS.snap_start_to_turn,
        )
        assert planned == direct

    def test_frozen_is_deterministic(self):
        s = _session()
        a, _ = plan_session(s, params=PARAMS, seed=0, epoch=None)
        b, _ = plan_session(s, params=PARAMS, seed=0, epoch=None)
        assert a == b


class TestEpochMode:
    def test_same_epoch_same_plan(self):
        s = _session()
        a, _ = plan_session(s, params=PARAMS, seed=0, epoch=3)
        b, _ = plan_session(s, params=PARAMS, seed=0, epoch=3)
        assert a == b

    def test_different_epochs_differ(self):
        s = _session()
        plans = [plan_session(s, params=PARAMS, seed=0, epoch=e)[0] for e in range(8)]
        spans = {tuple((w.t0, w.t1) for w in p) for p in plans}
        assert len(spans) > 1  # fresh windows across epochs

    def test_epoch_zero_differs_from_frozen(self):
        s = _session()
        frozen, _ = plan_session(s, params=PARAMS, seed=0, epoch=None)
        e0, _ = plan_session(s, params=PARAMS, seed=0, epoch=0)
        # Different RNG streams; identical plans would be astronomically
        # unlikely for this session unless the seed strings collided.
        assert [(w.t0, w.t1) for w in frozen] != [(w.t0, w.t1) for w in e0]


class TestAtomic:
    def test_atomic_passthrough(self):
        t = Turn(channel=0, speaker="sp", text="abc", start=0.0, end=3.2)
        s = _session(
            atomic=True,
            window_id="libritts_utt1",
            num_channels=1,
            turns=(t,),
            duration=3.2,
        )
        for epoch in (None, 0, 7):
            planned, _ = plan_session(s, params=PARAMS, seed=0, epoch=epoch)
            assert len(planned) == 1
            w = planned[0]
            assert w.window_id == "libritts_utt1"
            assert (w.t0, w.t1) == (0.0, 3.2)
            assert w.turns == (t,)

    def test_atomic_ignores_chunk_params(self):
        # BIT-PARITY: session.atomic must take the exact current code path
        # regardless of chunk_params - atomic records bypass planning
        # entirely, so passing a chunk_task_prob=1.0 must be a no-op.
        t = Turn(channel=0, speaker="sp", text="abc", start=0.0, end=3.2)
        s = _session(
            atomic=True,
            window_id="libritts_utt1",
            num_channels=1,
            turns=(t,),
            duration=3.2,
        )
        for epoch in (None, 0, 7):
            plain, _ = plan_session(s, params=PARAMS, seed=0, epoch=epoch)
            chunky, _ = plan_session(
                s,
                params=PARAMS,
                seed=0,
                epoch=epoch,
                chunk_params=ChunkTaskParams(chunk_task_prob=1.0),
            )
            assert plain == chunky


class TestExclusionSpans:
    def test_overlapping_windows_dropped(self):
        s = _session(exclusion_spans=((0.0, 1e9),))  # covers everything
        planned, stats = plan_session(s, params=PARAMS, seed=0, epoch=None)
        assert planned == []

    def test_non_overlapping_untouched(self):
        s0 = _session()
        s1 = _session(exclusion_spans=((1e6, 1e6 + 1),))
        a, _ = plan_session(s0, params=PARAMS, seed=0, epoch=None)
        b, _ = plan_session(s1, params=PARAMS, seed=0, epoch=None)
        assert a == b


def test_plan_sessions_concatenates_in_order():
    s1, s2 = _session(session_id="a"), _session(session_id="b")
    both, _ = plan_sessions([s1, s2], params=PARAMS, seed=0, epoch=None)
    only1, _ = plan_session(s1, params=PARAMS, seed=0, epoch=None)
    only2, _ = plan_session(s2, params=PARAMS, seed=0, epoch=None)
    assert both == only1 + only2


class TestChunkTaskPlanning:
    def test_prob_zero_is_bit_identical(self):
        # chunk_params=None (below) and chunk_task_prob=0.0 (here) must both
        # take the exact bit-parity path: zero extra rng draws, identical
        # output to the plain plan_session call.
        s = _session()
        base, _ = plan_session(s, params=WindowParams(), seed=3, epoch=5)
        same, _ = plan_session(
            s,
            params=WindowParams(),
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=0.0),
        )
        assert [to_json(r) for r in base] == [to_json(r) for r in same]

    def test_frozen_mode_never_chunks(self):
        # epoch=None (valid/test/inference) must never chunk regardless of
        # chunk_task_prob - part of the same bit-parity guarantee. Bit-compare
        # against the plain (chunk_params=None) call, not just chunk_task is
        # None: a gate that still draws the coin and discards it would pass a
        # weaker "chunk_task is None" check but fail this one.
        s = _session()
        plain, _ = plan_session(s, params=WindowParams(), seed=3, epoch=None)
        recs, _ = plan_session(
            s,
            params=WindowParams(),
            seed=3,
            epoch=None,
            chunk_params=ChunkTaskParams(chunk_task_prob=1.0),
        )
        assert all(r.chunk_task is None for r in recs)
        assert [to_json(r) for r in plain] == [to_json(r) for r in recs]

    def test_chunk_sessions_use_chunk_window_range_and_attach_plans(self):
        s = _session()
        recs, stats = plan_session(
            s,
            params=WindowParams(),
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=1.0, prompt_only_prob=0.0),
        )
        chunked = [r for r in recs if r.chunk_task is not None]
        assert chunked
        # Windows are planned against chunk_window_min/max (defaults 15/35),
        # not the base WindowParams() range (10/80).
        assert all(r.duration <= 35.0 + 1e-6 for r in chunked)
        # Every attached plan lands in exactly one of the two kind buckets;
        # this session's very first window always starts at session-start
        # (t0=0), so it structurally has no prefix material and always
        # degrades to prompt_only even with prompt_only_prob=0.0 - later
        # windows have real material behind them, so at least one is "full".
        assert stats.n_chunk_full + stats.n_chunk_prompt_only == len(chunked)
        # Hand-verified for this exact fixture/seed/epoch (window range
        # 15-35s over a 60s session yields exactly 2 windows): window 0 is
        # the structural degrade (t0=0, no session prefix), window 1 is a
        # genuine full plan. Exact values, not >=1, so a regression that
        # flips a window between full/degraded/dropped is caught.
        assert len(chunked) == 2
        assert stats.n_chunk_full == 1
        assert stats.n_chunk_prompt_only == 1
        assert stats.n_chunk_degraded == 1
        assert stats.n_chunk_fallback_infill == 0

    def test_rank_consistency(self):
        # Two independent plan_session calls with identical inputs (including
        # chunk_params) must agree exactly - required for DDP ranks/workers
        # that each re-derive windows independently.
        a, _ = plan_session(
            _session(),
            params=WindowParams(),
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=0.7),
        )
        b, _ = plan_session(
            _session(),
            params=WindowParams(),
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=0.7),
        )
        assert [to_json(r) for r in a] == [to_json(r) for r in b]

    def test_plan_sessions_merges_chunk_stats_and_threads_chunk_params(self):
        # Neither plan_sessions' merge() extension nor its chunk_params
        # passthrough is exercised by any single-session test above; a
        # dropped `+=` line or a dropped chunk_params kwarg in the loop body
        # would leave all-zero totals here while every other test still
        # passes (single-session calls don't touch plan_sessions at all).
        s1, s2 = _session(session_id="a"), _session(session_id="b")
        params = ChunkTaskParams(chunk_task_prob=1.0, prompt_only_prob=0.0)
        both, total = plan_sessions(
            [s1, s2], params=WindowParams(), seed=3, epoch=5, chunk_params=params
        )
        only1, stats1 = plan_session(
            s1, params=WindowParams(), seed=3, epoch=5, chunk_params=params
        )
        only2, stats2 = plan_session(
            s2, params=WindowParams(), seed=3, epoch=5, chunk_params=params
        )
        assert [to_json(r) for r in both] == [to_json(r) for r in only1 + only2]
        for field in (
            "n_chunk_full",
            "n_chunk_prompt_only",
            "n_chunk_degraded",
            "n_chunk_fallback_infill",
        ):
            merged = getattr(total, field)
            assert merged == getattr(stats1, field) + getattr(stats2, field)
        # This fixture's geometry never triggers a floor conflict, so
        # n_chunk_fallback_infill legitimately stays 0 for both sessions;
        # full/prompt_only/degraded are all non-zero per session (see
        # test_chunk_sessions_use_chunk_window_range_and_attach_plans), so
        # the merged totals must be too - catching a merge() no-op that the
        # equality check above alone would not (0 == 0 + 0 still passes it).
        assert total.n_chunk_full > 0
        assert total.n_chunk_prompt_only > 0
        assert total.n_chunk_degraded > 0

    def test_infill_branch_still_consumes_the_coin_draw(self):
        # chunk_task_prob=1e-9 makes is_chunk virtually certain to be False,
        # but chunk_params is not None / chunk_task_prob > 0 / epoch is not
        # None, so this still enters the CHUNKING branch, not the
        # bit-parity branch: the coin is drawn from the shared rng before
        # build_windows runs. Deleting the `if is_chunk:` guard around the
        # window-range override and draw_chunk_task calls would not be
        # caught by any bit-parity test above - those all use
        # chunk_task_prob == 0.0, which never even enters the chunking
        # branch, so they cannot see a stray coin draw either.
        s = _session()
        plain, _ = plan_session(s, params=PARAMS, seed=3, epoch=5)
        recs, _ = plan_session(
            s,
            params=PARAMS,
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=1e-9),
        )
        # (i) is_chunk came out False for this seed/session/epoch: nothing
        # got chunked.
        assert all(r.chunk_task is None for r in recs)
        # (ii) but the output still differs from the plain call: build_windows
        # here consumed an rng already advanced by one rng.random() call (the
        # coin draw), so its window boundaries diverge from the plain call's
        # untouched rng. Byte-identical output here would mean the coin was
        # never actually drawn from the shared rng. PARAMS (not the default
        # WindowParams(), whose 80s window_max exceeds this 60s session's
        # duration and so never calls rng.uniform at all) forces real cuts
        # that are sensitive to the extra draw.
        assert [to_json(r) for r in plain] != [to_json(r) for r in recs]

    def test_epoch_mode_parity_branch_pinned(self):
        # Golden-style pin for the bit-parity branch's output IN epoch mode
        # (the existing golden tests in test_parity.py only cover frozen
        # mode, epoch=None). Captured once at this commit by running the
        # exact call below and hand-copying its to_json output; any future
        # change to the parity branch's RNG construction or draw order (the
        # thing this task's bit-parity guarantee depends on) will change
        # this output and fail loudly here instead of silently.
        turns = (
            Turn(channel=0, speaker="s0", text="hello there", start=0.0, end=1.5),
            Turn(channel=1, speaker="s1", text="hello there", start=2.0, end=3.5),
            Turn(channel=0, speaker="s0", text="hello there", start=4.0, end=5.5),
            Turn(channel=1, speaker="s1", text="hello there", start=6.0, end=7.5),
            Turn(channel=0, speaker="s0", text="hello there", start=8.0, end=9.5),
            Turn(channel=1, speaker="s1", text="hello there", start=10.0, end=11.5),
        )
        s = _session(session_id="sess_golden", turns=turns, duration=12.0)
        params = WindowParams(window_min=3.0, window_max=5.0, tail_min=1.0)
        recs, _ = plan_session(s, params=params, seed=7, epoch=5, chunk_params=None)
        assert [to_json(r) for r in recs] == _EPOCH_MODE_GOLDEN

    def test_epoch_mode_parity_branch_pinned_matches_chunk_prob_zero(self):
        # Cross-check the pinned golden above against the chunk_task_prob==0
        # bit-parity path with the same inputs, so the golden constant and
        # the bit-parity guarantee are shown to agree, not just each
        # independently plausible.
        turns = (
            Turn(channel=0, speaker="s0", text="hello there", start=0.0, end=1.5),
            Turn(channel=1, speaker="s1", text="hello there", start=2.0, end=3.5),
            Turn(channel=0, speaker="s0", text="hello there", start=4.0, end=5.5),
            Turn(channel=1, speaker="s1", text="hello there", start=6.0, end=7.5),
            Turn(channel=0, speaker="s0", text="hello there", start=8.0, end=9.5),
            Turn(channel=1, speaker="s1", text="hello there", start=10.0, end=11.5),
        )
        s = _session(session_id="sess_golden", turns=turns, duration=12.0)
        params = WindowParams(window_min=3.0, window_max=5.0, tail_min=1.0)
        recs, _ = plan_session(
            s,
            params=params,
            seed=7,
            epoch=5,
            chunk_params=ChunkTaskParams(chunk_task_prob=0.0),
        )
        assert [to_json(r) for r in recs] == _EPOCH_MODE_GOLDEN

    def test_fallback_infill_counted_and_record_survives(self):
        # Lifts Task 3's cross-channel-floor-conflict geometry
        # (test_cross_channel_floor_conflict_returns_none) into a
        # planner-level fixture, engineered so the conflict is
        # order/rng-independent: window "b"'s channel-1 candidate pool has
        # exactly ONE eligible anchor (90.0-90.5s), and that anchor's
        # cumulative-speech search (_min_extent) only reaches the 3.0s floor
        # by walking forward into a LATER turn (115.0-118.0s) that sits
        # INSIDE the forbidden region - so m_c=27.5 while a_c=10.0
        # (m_c > a_c), failing the per-candidate check deterministically
        # regardless of which rng state reaches this draw. Window "a"
        # ([0, 100)) draws successfully (both channels have valid,
        # non-conflicting candidates outside its own forbidden region), so
        # exactly one of the two windows falls back.
        turns = (
            Turn(channel=1, speaker="B", text="short", start=90.0, end=90.5),
            Turn(channel=0, speaker="A", text="x", start=95.0, end=99.0),
            Turn(channel=0, speaker="A", text="y", start=110.0, end=113.0),
            Turn(channel=1, speaker="B", text="z", start=115.0, end=118.0),
        )
        s = _session(session_id="sess_fallback", turns=turns, duration=130.0)
        params = WindowParams(
            window_min=5.0,
            window_max=15.0,
            tail_min=1.0,
            trim_to_turns=False,
            snap_start_to_turn=False,
        )
        chunk_params = ChunkTaskParams(
            chunk_task_prob=1.0,
            prompt_only_prob=1.0,
            prompt_slice_min=1.0,
            prompt_slice_max=25.0,
            prompt_speech_floor=3.0,
            chunk_window_min=100.0,
            chunk_window_max=100.0,
        )
        recs, stats = plan_session(
            s, params=params, seed=11, epoch=5, chunk_params=chunk_params
        )
        assert [(r.t0, r.t1) for r in recs] == [(0.0, 100.0), (100.0, 130.0)]
        window_a, window_b = recs
        assert window_a.chunk_task is not None  # succeeds: candidates outside
        assert window_b.chunk_task is None  # falls back: infill, not dropped
        assert stats.n_chunk_fallback_infill == 1
        assert stats.n_chunk_prompt_only == 1
        assert stats.n_chunk_full == 0
        assert stats.n_chunk_degraded == 0


class TestTimestampCoin:
    def test_prob_zero_is_bit_parity(self):
        s = _session()
        base, base_stats = plan_session(s, params=PARAMS, seed=0, epoch=3)
        same, same_stats = plan_session(
            s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=0.0
        )
        assert base == same
        # The coin's only observable side channel is the counters (Mode T
        # uses its own rng stream, so a gate bug that still draws-and-skips
        # would leave records untouched but pollute these) - weaker checks
        # that only compare records would miss it, per the file's own
        # test_frozen_mode_never_chunks convention.
        assert base_stats.n_timestamp_windows == 0
        assert base_stats.n_timestamp_degraded == 0
        assert same_stats.n_timestamp_windows == 0
        assert same_stats.n_timestamp_degraded == 0

    def test_coin_changes_flags_not_windows(self):
        s = _session()
        base, _ = plan_session(s, params=PARAMS, seed=0, epoch=3)
        flagged, stats = plan_session(
            s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        assert [(r.t0, r.t1, r.chunk_task) for r in base] == [
            (r.t0, r.t1, r.chunk_task) for r in flagged
        ]
        assert stats.n_timestamp_windows + stats.n_timestamp_degraded == len(flagged)
        assert all(
            r.timestamp_text == fits
            for r, fits in zip(
                flagged,
                [timestamp_fits(r.turns, r.t0, r.t1) for r in base],
            )
        )

    def test_frozen_mode_never_flags(self):
        s = _session()
        records, stats = plan_session(
            s, params=PARAMS, seed=0, epoch=None, timestamp_align_prob=1.0
        )
        assert not any(r.timestamp_text for r in records)
        # Same weaker-check trap as above: epoch=None must never even draw
        # the coin, not just discard a heads result.
        assert stats.n_timestamp_windows == 0
        assert stats.n_timestamp_degraded == 0

    def test_deterministic_across_calls(self):
        s = _session()
        a, _ = plan_session(s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=0.5)
        b, _ = plan_session(s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=0.5)
        assert [r.timestamp_text for r in a] == [r.timestamp_text for r in b]

    def test_atomic_path_can_flag(self):
        # BIT-PARITY routing check: the atomic early return must also go
        # through the coin, not just the chunking/non-chunking tails.
        t = Turn(channel=0, speaker="sp", text="hi", start=0.0, end=3.2)
        s = _session(
            atomic=True, window_id="w1", num_channels=1, turns=(t,), duration=3.2
        )
        records, stats = plan_session(
            s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        assert records[0].timestamp_text is True
        assert stats.n_timestamp_windows == 1
        assert stats.n_timestamp_degraded == 0

    def test_atomic_path_can_degrade(self):
        # One sub-0.05 s turn carrying far more text than its frame span can
        # hold: timestamp_fits must reject it, so a coin-heads draw degrades
        # back to Mode O instead of flagging timestamp_text.
        t = Turn(channel=0, speaker="sp", text="x" * 200, start=0.0, end=0.03)
        s = _session(
            atomic=True, window_id="w1", num_channels=1, turns=(t,), duration=0.03
        )
        records, stats = plan_session(
            s, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        assert stats.n_timestamp_degraded >= 1
        assert records[0].timestamp_text is False

    def test_plan_sessions_forwards_prob_and_merges_stats(self):
        # Mirrors test_plan_sessions_merges_chunk_stats_and_threads_chunk_params:
        # a dropped timestamp_align_prob kwarg in the plan_sessions loop body
        # would leave both counters at 0 while every single-session test
        # above still passes untouched.
        s1, s2 = _session(session_id="a"), _session(session_id="b")
        both, total = plan_sessions(
            [s1, s2], params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        only1, stats1 = plan_session(
            s1, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        only2, stats2 = plan_session(
            s2, params=PARAMS, seed=0, epoch=3, timestamp_align_prob=1.0
        )
        assert both == only1 + only2
        assert total.n_timestamp_windows == (
            stats1.n_timestamp_windows + stats2.n_timestamp_windows
        )
        assert total.n_timestamp_degraded == (
            stats1.n_timestamp_degraded + stats2.n_timestamp_degraded
        )
        assert total.n_timestamp_windows > 0


class TestMaskCoin:
    def test_zero_probs_never_touch_the_stream(self, monkeypatch):
        from ..preprocessing import planner as planner_mod

        def _boom(*a, **kw):
            raise AssertionError("maskmode rng must not be constructed")

        monkeypatch.setattr(planner_mod, "_maskmode_rng", _boom)
        records, _ = plan_session(
            _session(), params=PARAMS, seed=0, epoch=3
        )
        assert all(r.context_channels is None for r in records)
        assert all(not r.independent_mask for r in records)

    def test_frozen_mode_never_flags(self):
        records, stats = plan_session(
            _session(),
            params=PARAMS,
            seed=0,
            epoch=None,
            context_channel_prob=1.0,
            independent_mask_prob=1.0,
        )
        assert all(r.context_channels is None for r in records)
        assert all(not r.independent_mask for r in records)
        assert stats.n_context_windows == 0
        assert stats.n_independent_windows == 0

    def test_context_prob_one_flags_every_window(self):
        records, stats = plan_session(
            _session(),
            params=PARAMS,
            seed=0,
            epoch=2,
            context_channel_prob=1.0,
            independent_mask_prob=1.0,
        )
        assert records
        for r in records:
            # N=2 windows: k ~ U{1..1}, a proper nonempty subset.
            assert r.context_channels is not None
            assert len(r.context_channels) == 1
            assert r.context_channels[0] in (0, 1)
            # Layered coin: a context hit never also flags independent.
            assert r.independent_mask is False
        assert stats.n_context_windows == len(records)
        assert stats.n_independent_windows == 0

    def test_independent_prob_one_flags_every_window(self):
        records, stats = plan_session(
            _session(),
            params=PARAMS,
            seed=0,
            epoch=2,
            context_channel_prob=0.0,
            independent_mask_prob=1.0,
        )
        assert records
        assert all(r.independent_mask for r in records)
        assert all(r.context_channels is None for r in records)
        assert stats.n_independent_windows == len(records)

    def test_mask_stream_does_not_perturb_windows(self):
        plain, _ = plan_session(_session(), params=PARAMS, seed=0, epoch=2)
        flagged, _ = plan_session(
            _session(),
            params=PARAMS,
            seed=0,
            epoch=2,
            context_channel_prob=1.0,
        )

        def strip(d):
            d = dict(d)
            d.pop("context_channels", None)
            d.pop("independent_mask", None)
            return d
        assert [strip(to_json(r)) for r in flagged] == [
            to_json(r) for r in plain
        ]

    def test_deterministic_across_calls(self):
        a, _ = plan_session(
            _session(), params=PARAMS, seed=0, epoch=2,
            context_channel_prob=0.5, independent_mask_prob=0.5,
        )
        b, _ = plan_session(
            _session(), params=PARAMS, seed=0, epoch=2,
            context_channel_prob=0.5, independent_mask_prob=0.5,
        )
        assert [to_json(r) for r in a] == [to_json(r) for r in b]

    def test_single_channel_context_hit_degrades_to_independent(self):
        rec = WindowRecord(
            window_id="w",
            session_id="s",
            audio_relpath="a.flac",
            num_channels=1,
            sample_rate=48000,
            t0=0.0,
            t1=3.0,
            turns=(
                Turn(channel=0, speaker="s0", text="hi", start=0.0, end=1.0),
            ),
        )
        stats = WindowingStats()
        out = _apply_mask_coin(
            [rec],
            stats,
            SimpleNamespace(session_id="s"),
            0,
            1,
            1.0,
            1.0,
        )
        assert out[0].context_channels is None
        assert out[0].independent_mask is True
        assert stats.n_context_degraded == 1
        assert stats.n_independent_windows == 1
        assert stats.n_context_windows == 0

    def test_context_subset_draw_at_n3(self):
        rec = WindowRecord(
            window_id="w",
            session_id="s",
            audio_relpath="a.flac",
            num_channels=3,
            sample_rate=48000,
            t0=0.0,
            t1=3.0,
            turns=(
                Turn(channel=0, speaker="s0", text="hi", start=0.0, end=1.0),
                Turn(channel=1, speaker="s1", text="yo", start=1.0, end=2.0),
                Turn(channel=2, speaker="s2", text="hey", start=2.0, end=3.0),
            ),
        )
        stats = WindowingStats()
        out = _apply_mask_coin(
            [rec],
            stats,
            SimpleNamespace(session_id="s"),
            0,
            1,
            1.0,
            0.0,
        )
        chans = out[0].context_channels
        assert 1 <= len(chans) <= 2
        assert all(c in range(3) for c in chans)
        assert list(chans) == sorted(set(chans))
        assert out[0].independent_mask is False

    def test_mask_coin_composes_with_chunk_task(self):
        # Same fixture/seed/epoch/params as TestChunkTaskPlanning's
        # test_chunk_sessions_use_chunk_window_range_and_attach_plans, which
        # is hand-verified to attach chunk plans to 2 windows.
        records, _ = plan_session(
            _session(),
            params=WindowParams(),
            seed=3,
            epoch=5,
            chunk_params=ChunkTaskParams(
                chunk_task_prob=1.0, prompt_only_prob=0.0
            ),
            context_channel_prob=1.0,
        )
        assert any(
            r.chunk_task is not None and r.context_channels is not None
            for r in records
        )
