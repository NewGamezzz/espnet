"""``TurnTakingJudgeMetric`` (``src/metrics/turn_taking_judge.py``) tests.

Pure units first (grid, rasteriser, backchannel proxy, upstream label state
machine, role swap), then the judge loop with an injected ``encode_fn``, then
the metric end to end with a fake VAD and an oracle judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    CHUNK,
    MIN_START,
    apply_backchannel_proxy,
    chunk_ends,
    label_rows,
    rasterise_channel,
    swap_roles,
)


# --------------------------------------------------------------------------- #
# Task 1: grid + rasteriser
# --------------------------------------------------------------------------- #
class TestGrid:
    def test_chunk_ends_match_upstream_grid(self):
        ends = chunk_ends(1.0)
        # int((1.0 - 0.2) / 0.04) = 20 chunks, ends 0.24 .. 1.00
        assert len(ends) == 20
        assert ends[0] == 0.24 and ends[-1] == 1.0
        assert all(isinstance(e, float) for e in ends)

    def test_chunk_ends_are_two_decimal_floats(self):
        ends = chunk_ends(24.0)
        assert ends == [
            float(f"{MIN_START + (i + 1) * CHUNK:.2f}") for i in range(len(ends))
        ]

    def test_rasterise_uses_upstream_inclusion_rule(self):
        ends = chunk_ends(1.0)  # 0.24 .. 1.00
        labels = rasterise_channel([(0.40, 0.60)], ends)
        # active iff s - 0.04 <= e < t: ends 0.36, 0.40, 0.44, 0.48, 0.52, 0.56
        active = [e for e, lab in zip(ends, labels) if lab == "IPU"]
        assert active == [0.36, 0.4, 0.44, 0.48, 0.52, 0.56]

    def test_rasterise_empty_is_all_silence(self):
        assert set(rasterise_channel([], chunk_ends(0.5))) == {"NA"}


# --------------------------------------------------------------------------- #
# Task 2: backchannel proxy
# --------------------------------------------------------------------------- #
class TestBackchannelProxy:
    @staticmethod
    def _kinds(out, ch):
        return [k for _, k in out[ch]]

    def test_short_span_inside_other_ipu_is_bc(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(2.0, 2.5)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]
        assert self._kinds(out, 0) == ["IPU"]

    def test_short_span_in_floor_holders_pause_is_bc(self):
        # ch0 holds the floor 0-2 and resumes 2.8-5; ch1's 2.1-2.5 is a BC.
        out = apply_backchannel_proxy([[(0.0, 2.0), (2.8, 5.0)], [(2.1, 2.5)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]

    def test_short_span_that_takes_the_floor_stays_ipu(self):
        # ch0 never speaks again before ch1's next IPU -> ch1 took the turn.
        out = apply_backchannel_proxy([[(0.0, 2.0)], [(2.1, 2.5), (3.0, 5.0)]], 6.0)
        assert self._kinds(out, 1) == ["IPU", "IPU"]

    def test_span_longer_than_cap_stays_ipu(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(1.0, 2.2)]], 6.0)
        assert self._kinds(out, 1) == ["IPU"]

    def test_cap_is_inclusive_at_1_08(self):
        out = apply_backchannel_proxy([[(0.0, 5.0)], [(1.0, 2.08)]], 6.0)
        assert self._kinds(out, 1) == ["BC"]

    def test_first_utterance_of_the_window_is_never_bc(self):
        out = apply_backchannel_proxy([[(0.0, 0.5)], [(1.0, 5.0)]], 6.0)
        assert self._kinds(out, 0) == ["IPU"]

    def test_non_two_channel_input_is_all_ipu(self):
        out = apply_backchannel_proxy([[(0.0, 0.5)]], 6.0)
        assert self._kinds(out, 0) == ["IPU"]


# --------------------------------------------------------------------------- #
# Task 3: label rows (upstream state machine) + role swap
# --------------------------------------------------------------------------- #
def _labels(rows):
    return [r.split(",")[3] for r in rows]


def _turns(rows):
    return [r.split(",")[4] for r in rows]


class TestLabelRows:
    def test_row_format_and_grid(self):
        rows = label_rows("w", [[], []], 1.0)
        assert len(rows) == 20
        assert rows[0] == "w,0.2,0.24,NA,NA"
        assert rows[-1].startswith("w,0.96,1.0,")

    def test_turn_change_then_continuation(self):
        rows = label_rows("w", [[(0.2, 0.6)], [(0.8, 1.2)]], 1.4)
        labels = _labels(rows)
        first_ch0 = labels.index("T")
        assert labels[first_ch0 + 1] == "C"
        assert "NA" in labels[first_ch0 + 1 :]
        assert labels.count("T") == 2
        turns = _turns(rows)
        assert turns[0] == "NA" and "A" in turns and turns[-1] == "B"

    def test_pause_by_same_speaker_is_continuation_not_turn(self):
        rows = label_rows("w", [[(0.2, 0.6), (0.9, 1.3)], []], 1.4)
        assert _labels(rows).count("T") == 1

    def test_overlap_is_interruption_and_floor_passes(self):
        # ch0 0.2-1.0, ch1 0.6-1.8 (too long for the BC proxy): I during the
        # overlap, then ch1 alone -> T because the floor passes from AB to B.
        rows = label_rows("w", [[(0.2, 1.0)], [(0.6, 1.8)]], 2.0)
        labels = _labels(rows)
        assert "I" in labels
        i_last = len(labels) - 1 - labels[::-1].index("I")
        assert labels[i_last + 1] == "T"
        assert "AB" in _turns(rows)

    def test_interrupted_speaker_continuing_alone_is_c(self):
        rows = label_rows("w", [[(0.2, 2.0)], [(0.6, 1.8)]], 2.2)
        labels = _labels(rows)
        i_last = len(labels) - 1 - labels[::-1].index("I")
        assert labels[i_last + 1] == "C"

    def test_backchannel_inside_other_ipu(self):
        rows = label_rows("w", [[(0.2, 3.0)], [(1.0, 1.4)]], 3.2)
        labels = _labels(rows)
        assert "BC" in labels and "I" not in labels

    def test_bc_in_floor_holders_pause_is_plain_bc(self):
        # ch1's short span sits in ch0's pause and ch0 resumes: proxy -> BC,
        # state machine prev speaker A -> plain BC (never the BC_1 exception).
        rows = label_rows("w", [[(0.2, 1.0), (1.6, 3.0)], [(1.1, 1.4)]], 3.2)
        labels = _labels(rows)
        assert "BC" in labels and "BC_1" not in labels

    def test_single_channel_window_is_padded_with_silence(self):
        rows = label_rows("w", [[(0.2, 0.6)]], 1.0)
        assert _labels(rows).count("T") == 1 and "B" not in _turns(rows)

    def test_three_channels_raise(self):
        with pytest.raises(ValueError):
            label_rows("w", [[], [], []], 1.0)


class TestSwapRoles:
    def test_swaps_turn_column_only(self):
        rows = [
            "w,0.2,0.24,C,A",
            "w,0.24,0.28,I,AB",
            "w,0.28,0.32,NA,NA",
            "w,0.32,0.36,T,B",
        ]
        assert swap_roles(rows) == [
            "w,0.2,0.24,C,B",
            "w,0.24,0.28,I,BA",
            "w,0.28,0.32,NA,NA",
            "w,0.32,0.36,T,A",
        ]
