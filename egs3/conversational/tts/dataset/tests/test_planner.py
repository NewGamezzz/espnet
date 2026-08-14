import random

from ..preprocessing.planner import WindowParams, plan_session, plan_sessions
from ..preprocessing.sessions import SessionRecord
from ..preprocessing.sssd import Recording, Turn
from ..preprocessing.windows import build_windows

# Long two-channel session with clean alternation and real gaps so several
# windows fit (window params shrunk so the 60 s session yields multiple).
PARAMS = WindowParams(window_min=5.0, window_max=15.0, tail_min=2.0)


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
