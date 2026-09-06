"""src/loudness.py: active-RMS level, floor-derived threshold, gain with
peak cap, and the gate sidecar's floor table."""
import json

import numpy as np
import pytest

from egs3.conversational.tts.src.loudness import (
    DEFAULT_ACTIVE_THRESHOLD,
    active_rms_db,
    db_to_lin,
    gain_to_target,
    load_channel_floors,
    loudness_meta,
    threshold_from_floor,
)

SR = 24000


def _tone(amp, seconds=2.0, hz=300):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TestLevel:
    def test_sine_level_is_rms(self):
        # 0.2 sine: RMS 0.1414 = -17.0 dBFS
        assert active_rms_db(_tone(0.2), SR) == pytest.approx(-17.0, abs=0.1)

    def test_silence_is_none(self):
        assert active_rms_db(np.zeros(SR), SR) is None
        assert active_rms_db(np.zeros(10), SR) is None  # shorter than one frame

    def test_noise_floor_above_default_threshold_is_excluded_by_floor_margin(self):
        # -45 dBFS noise everywhere + a -20 dBFS tone for the first half.
        rng = np.random.default_rng(0)
        x = rng.normal(0, db_to_lin(-45), SR * 2).astype(np.float32)
        x[: SR] += _tone(db_to_lin(-20) * np.sqrt(2), 1.0)
        low = active_rms_db(x, SR, DEFAULT_ACTIVE_THRESHOLD)   # noise frames count
        high = active_rms_db(x, SR, threshold_from_floor(-45.0, 10.0))
        assert low < high - 2.0          # the default under-reads
        assert high == pytest.approx(-20.0, abs=0.3)


class TestThreshold:
    def test_from_floor(self):
        assert threshold_from_floor(None, 10.0) == DEFAULT_ACTIVE_THRESHOLD
        assert threshold_from_floor(-90.0, 10.0) == DEFAULT_ACTIVE_THRESHOLD  # never below
        assert threshold_from_floor(-50.0, 10.0) == pytest.approx(db_to_lin(-40.0))


class TestGain:
    def test_gain_hits_target(self):
        g, limited, level = gain_to_target(_tone(0.2), SR, -23.0)
        assert level == pytest.approx(-17.0, abs=0.1)
        assert not limited
        assert active_rms_db(_tone(0.2) * g, SR) == pytest.approx(-23.0, abs=0.1)

    def test_peak_cap_uses_the_supplied_peak(self):
        g, limited, _ = gain_to_target(_tone(0.2), SR, 0.0, peak=0.5)
        assert limited and g == pytest.approx(0.99 / 0.5)

    def test_silence_passes_through(self):
        assert gain_to_target(np.zeros(SR), SR, -23.0) == (1.0, False, None)


class TestSidecar:
    def test_floors_loaded_and_missing_key_tolerated(self, tmp_path):
        p = tmp_path / "ex.json"
        p.write_text(json.dumps({"version": 1, "spans": [], "floor_db": {"s": [-50.0, None]}}))
        assert load_channel_floors(p) == {"s": [-50.0, None]}
        p.write_text(json.dumps({"version": 1, "spans": []}))
        assert load_channel_floors(p) == {}
        p.write_text(json.dumps({"version": 2, "spans": []}))
        with pytest.raises(ValueError):
            load_channel_floors(p)

    def test_meta_shape(self):
        assert loudness_meta(None, None, None, None, None) is None
        m = loudness_meta(-23.0, [1e-3], [0.5], [False], [-17.0])
        assert m == {"target_db": -23.0, "threshold_db": [-60.0], "level_db": [-17.0],
                     "gain_db": [pytest.approx(-6.021, abs=1e-3)], "peak_limited": [False]}
