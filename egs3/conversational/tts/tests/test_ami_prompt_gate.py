"""local/ami_prompt_gate.py: silence floors, solo-by-annotation guard, the
own-floor energy gate and the bleed table - on synthetic 4-channel audio."""
import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.local.ami_prompt_gate import (
    bleed_rows,
    channel_floor_db,
    gate_session,
    silence_regions,
    solo_by_annotation,
    turn_excess_db,
)

SR = 24000


def _audio(seconds=30.0, bleed_at=None):
    """Four channels of near-silent noise (-60 dBFS); channel 1 speaks at
    5-8 s and channel 2 at 12-15 s (-20 dBFS tones); ``bleed_at=(k, s, e)``
    adds a -26 dBFS tone on channel k there."""
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    x = rng.normal(0, 1e-3, size=(n, 4)).astype("float32")
    t = np.arange(n) / SR
    x[int(5 * SR) : int(8 * SR), 1] += 0.1 * np.sin(2 * np.pi * 220 * t[int(5 * SR) : int(8 * SR)])
    x[int(12 * SR) : int(15 * SR), 2] += 0.1 * np.sin(
        2 * np.pi * 260 * t[int(12 * SR) : int(15 * SR)]
    )
    if bleed_at:
        k, s, e = bleed_at
        x[int(s * SR) : int(e * SR), k] += 0.05 * np.sin(2 * np.pi * 330 * t[int(s * SR) : int(e * SR)])
    return x


TURNS = [Turn(1, "b", "one two three", 5.0, 8.0), Turn(2, "c", "four five six", 12.0, 15.0)]


class TestRegions:
    def test_silence_regions_exclude_turns_with_guard(self):
        regs = silence_regions(TURNS, duration=30.0, guard=0.5)
        assert regs == [(0.0, 4.5), (8.5, 11.5), (15.5, 30.0)]

    def test_solo_by_annotation_guard(self):
        turns = TURNS + [Turn(0, "a", "yeah", 8.1, 8.4)]
        assert solo_by_annotation(turns, TURNS[0], guard=0.0)
        assert not solo_by_annotation(turns, TURNS[0], guard=0.3)


class TestLevels:
    def test_floor_and_excess(self):
        x = _audio(bleed_at=(3, 5.5, 7.5))
        floors = channel_floor_db(x, silence_regions(TURNS, 30.0, 0.5), SR)
        assert all(-62 < f < -58 for f in floors)
        excess = turn_excess_db(x, TURNS[0], SR, floors)
        assert excess[1] > 30  # own channel, not gated
        assert excess[3] > 20  # bleed on channel 3
        assert excess[0] < 3 and excess[2] < 3


class TestGate:
    def _flac(self, tmp_path, x):
        path = tmp_path / "m.flac"
        sf.write(str(path), x, SR, subtype="PCM_16", format="FLAC")
        return path

    def test_gate_session_excludes_bleed_and_non_solo(self, tmp_path):
        path = self._flac(tmp_path, _audio(bleed_at=(3, 5.5, 7.5)))
        turns = TURNS + [Turn(0, "a", "yeah", 14.9, 15.2)]
        result = gate_session(
            "m", path, turns, duration=30.0, turn_min=2.0, turn_max=10.0,
            solo_guard=0.3, max_excess_db=6.0, silence_guard=0.5,
        )
        reasons = {(s["channel"], s["start"], s["end"]): s["reason"] for s in result["excluded"]}
        assert reasons[(1, 5.0, 8.0)] == "energy:ch3"
        assert reasons[(2, 12.0, 15.0)] == "not_solo"
        assert len(result["floor_db"]) == 4
        assert result["n_candidates"] == 2 and result["n_accepted"] == 0
        assert all(s["session_id"] == "m" for s in result["excluded"])

    def test_gate_session_accepts_clean_solo(self, tmp_path):
        path = self._flac(tmp_path, _audio())
        result = gate_session(
            "m", path, TURNS, duration=30.0, turn_min=2.0, turn_max=10.0,
            solo_guard=0.3, max_excess_db=6.0, silence_guard=0.5,
        )
        assert result["excluded"] == []
        assert result["n_accepted"] == 2
        assert len(result["accepted_worst_excess_db"]) == 2

    def test_bleed_rows_measure_other_channels_over_solo_regions(self, tmp_path):
        path = self._flac(tmp_path, _audio(bleed_at=(3, 5.5, 7.5)))
        rows = bleed_rows("m", path, TURNS, guard=0.2)
        # (session, channel k, speaking channel j, solo_sec, bleed_db, own_db)
        by = {(r[1], r[2]): r for r in rows}
        assert set(by) == {(k, j) for j in (1, 2) for k in range(4) if k != j}
        assert by[(3, 1)][4] > by[(0, 1)][4] + 15  # bleed on ch3 while ch1 speaks
        assert by[(0, 2)][4] < -30  # clean channel while ch2 speaks
