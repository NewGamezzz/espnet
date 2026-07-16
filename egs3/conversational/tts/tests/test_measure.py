"""Measure-stage round trip: a fabricated ``inference_dir`` with ``meta.scp``
driven through ``espnet3.systems.base.metric.measure`` (used as-is) by a stub
``BaseMetric`` that iterates ``meta.scp`` as its single input, proving the
``inputs: {meta: meta}`` shape every real metric (later tasks) will use.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from egs3.conversational.tts.src.system import ConversationalTTSSystem
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.base.metric import measure


class StubMetaMetric(BaseMetric):
    """Minimal metric proving the meta.scp round trip: opens each window's
    meta JSON (path relative to the test-set dir, per the infer-stage
    contract) and summarizes channel counts."""

    def __call__(self, data, test_name, output_dir):
        n_windows = 0
        total_channels = 0
        seen_ids = []
        test_dir = Path(data["meta"]).parent
        for window_id, row in self.iter_inputs(data, "meta"):
            meta = json.loads((test_dir / row["meta"]).read_text("utf-8"))
            n_windows += 1
            total_channels += meta["num_channels"]
            seen_ids.append(window_id)
        return {
            "n_windows": n_windows,
            "total_channels": total_channels,
            "window_ids": seen_ids,
        }


def _write_meta_fixture(test_dir: Path, window_ids: list[str]) -> None:
    """A fabricated inference_dir/<test_name>/ tree: meta.scp + meta JSONs
    shaped like the infer stage's real output contract (src/inference.py)."""
    (test_dir / "meta").mkdir(parents=True)
    meta_lines = []
    for i, wid in enumerate(window_ids):
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "gt",
            "sample_rate": 24000,
            "num_channels": 2,
            "window_duration_sec": 12.0,
            "rtf": None,
            "mix_wav": f"mix/{wid}.wav",
            "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "hello",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "world",
                },
            ],
            "turns": [],
        }
        meta_rel = f"meta/{wid}.json"
        (test_dir / meta_rel).write_text(json.dumps(meta), encoding="utf-8")
        meta_lines.append(f"{wid} {meta_rel}")
    (test_dir / "meta.scp").write_text(
        "".join(f"{line}\n" for line in meta_lines), encoding="utf-8"
    )


def _metrics_config(inference_dir: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": "valid"}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.StubMetaMetric"},
                    "inputs": {"meta": "meta"},
                }
            ],
        }
    )


class TestMeasureRoundTrip:
    def test_writes_metrics_json_from_fabricated_inference_dir(self, tmp_path):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_meta_fixture(test_dir, ["sess_w00000", "sess_w00001"])

        cfg = _metrics_config(inference_dir)
        results = measure(cfg)

        from espnet3.utils.scp_utils import get_class_path

        key = get_class_path(StubMetaMetric())
        assert results[key]["valid"] == {
            "n_windows": 2,
            "total_channels": 4,
            "window_ids": ["sess_w00000", "sess_w00001"],
        }

        metrics_path = inference_dir / "metrics.json"
        assert metrics_path.is_file()
        assert json.loads(metrics_path.read_text("utf-8")) == results

    def test_meta_paths_resolve_relative_to_the_test_dir(self, tmp_path):
        """meta.scp rows and the meta JSON's own paths are relative to
        inference_dir/<test_name>/, per src/inference.py's output contract;
        StubMetaMetric resolving them via ``Path(data["meta"]).parent`` must
        actually find the files (not just parse the JSON structurally)."""
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_meta_fixture(test_dir, ["sess_w00000"])

        cfg = _metrics_config(inference_dir)
        results = measure(cfg)

        from espnet3.utils.scp_utils import get_class_path

        key = get_class_path(StubMetaMetric())
        assert results[key]["valid"]["n_windows"] == 1

    def test_system_measure_dispatches_to_the_same_machinery(self, tmp_path):
        """The production path: ConversationalTTSSystem.measure(), which is
        inherited unmodified from BaseSystem."""
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        _write_meta_fixture(test_dir, ["sess_w00000"])

        cfg = _metrics_config(inference_dir)
        system = ConversationalTTSSystem(metrics_config=cfg)
        results = system.measure()

        from espnet3.utils.scp_utils import get_class_path

        key = get_class_path(StubMetaMetric())
        assert results[key]["valid"]["n_windows"] == 1
        assert (inference_dir / "metrics.json").is_file()
