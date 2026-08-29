"""``local/turn_judge_bootstrap.py``: point estimates must reproduce the
metric's own summary, CIs must bracket them, and differences are paired."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    TurnTakingJudge,
    TurnTakingJudgeMetric,
)

RECIPE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "turn_judge_bootstrap", RECIPE / "local" / "turn_judge_bootstrap.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _VAD:
    def __call__(self, wav, sr):
        n = len(wav) / sr
        s = float(abs(wav[:10]).sum()) % 0.7
        return [(0.2 + s, 1.0 + s), (2.0 + s, min(n - 0.2, 2.6 + s))]


def _score_run(root: Path, run: str, seed: int):
    rng = np.random.default_rng(seed)
    td = root / run / "valid"
    (td / "meta").mkdir(parents=True)
    scp = []
    for k in range(10):
        dur, sr = 4.0 + k * 0.3, 16000
        chans = []
        for ch in range(2):
            rel = f"wav/w{k}_ch{ch}.wav"
            (td / "wav").mkdir(exist_ok=True)
            sf.write(td / rel, (rng.standard_normal(int(dur * sr)) * 0.1).astype(np.float32), sr)
            chans.append({"gen_wav": rel})
        (td / "mix").mkdir(exist_ok=True)
        sf.write(td / f"mix/w{k}.wav", (rng.standard_normal(int(dur * sr)) * 0.1).astype(np.float32), sr)
        (td / f"meta/w{k}.json").write_text(
            json.dumps({"window_id": f"w{k}", "window_duration_sec": dur, "mix_wav": f"mix/w{k}.wav", "channels": chans})
        )
        scp.append(f"w{k} meta/w{k}.json")
    (td / "meta.scp").write_text("\n".join(scp) + "\n")

    def enc(batch):
        return rng.dirichlet(np.ones(5) * 2, size=len(batch)).astype(np.float32)

    metric = TurnTakingJudgeMetric(
        judge=TurnTakingJudge(encode_fn=enc), vad_backend=_VAD(), report_role_metrics=True
    )
    return metric({"meta": td / "meta.scp"}, "valid", root / run), td


class TestBootstrap:
    def test_point_matches_metric_and_cis_bracket(self, tmp_path, monkeypatch, capsys):
        mod = _load()
        summary_a, td_a = _score_run(tmp_path, "a", 0)
        summary_b, td_b = _score_run(tmp_path, "b", 1)
        monkeypatch.setattr(mod.sys, "argv", ["x", "12", "3", f"a={td_a}", f"b={td_b}"])
        mod.main()
        out = json.loads((tmp_path / "judge_bootstrap_a_b.json").read_text())
        assert out["n_windows"] == 10 and out["n_reps"] == 12
        for k, v in summary_a.items():
            if k in out["point"]["a"] and v is not None:
                assert out["point"]["a"][k] == pytest.approx(v, abs=1e-9), k
                lo, hi = out["ci"]["a"][k]
                assert lo - 1e-9 <= v <= hi + 1e-9, k
        d = out["diff"]["a - b"]["judge_f1_macro"]
        assert d["point"] == pytest.approx(summary_a["judge_f1_macro"] - summary_b["judge_f1_macro"])
        assert set(out["counts"]["a"]) <= set(mod.ROLE_ORDER)
        assert "| f1_macro |" in capsys.readouterr().out
