"""``local/measure_merge.py``: measure() overwrites metrics.json; the merge
runner must keep the metrics that were already there."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric

RECIPE = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "measure_merge", RECIPE / "local" / "measure_merge.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CountingStubMetric(BaseMetric):
    def __call__(self, data, test_name, output_dir):
        n = sum(1 for _ in self.iter_inputs(data, "meta"))
        return {"n_windows": n}


class TestMergeMetrics:
    def test_merge_keeps_existing_keys_and_backs_up(self, tmp_path):
        mm = _load_module()
        (tmp_path / "metrics.json").write_text(json.dumps({"old.Metric": {"valid": {"wer": 0.1}}}))
        merged = mm.merge_metrics(tmp_path, {"new.Metric": {"valid": {"judge_f1_macro": 0.5}}})
        assert merged == {
            "old.Metric": {"valid": {"wer": 0.1}},
            "new.Metric": {"valid": {"judge_f1_macro": 0.5}},
        }
        assert json.loads((tmp_path / "metrics.json").read_text()) == merged
        assert json.loads((tmp_path / "metrics.json.bak").read_text()) == {
            "old.Metric": {"valid": {"wer": 0.1}}
        }

    def test_run_merges_into_existing_metrics_json(self, tmp_path):
        mm = _load_module()
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        (test_dir / "meta").mkdir(parents=True)
        (test_dir / "meta" / "w1.json").write_text("{}")
        (test_dir / "meta.scp").write_text("w1 meta/w1.json\n")
        (inference_dir / "metrics.json").write_text(json.dumps({"old.Metric": {"valid": {"wer": 0.1}}}))
        cfg = OmegaConf.create(
            {
                "inference_dir": str(inference_dir),
                "dataset": {"test": [{"name": "valid"}]},
                "metrics": [
                    {
                        "metric": {"_target_": "tests.test_measure_merge.CountingStubMetric"},
                        "inputs": {"meta": "meta"},
                    },
                    {
                        "metric": {"_target_": "tests.test_measure_merge.CountingStubMetric"},
                        "inputs": {"meta": "meta"},
                    },
                ],
            }
        )
        cfg_path = tmp_path / "metrics.yaml"
        OmegaConf.save(cfg, cfg_path)
        merged = mm.run(cfg_path, only="CountingStubMetric")
        assert merged["old.Metric"] == {"valid": {"wer": 0.1}}
        stub_key = [k for k in merged if k.endswith("CountingStubMetric")][0]
        assert merged[stub_key] == {"valid": {"n_windows": 1}}
        assert json.loads((inference_dir / "metrics.json").read_text()) == merged


    def test_merge_is_key_level_within_a_metric_class(self, tmp_path):
        from egs3.conversational.tts.local.measure_merge import merge_metrics

        cls = "x.TurnTakingJudgeMetric"
        merge_metrics(tmp_path, {cls: {"valid": {"judge_f1_macro": 0.5, "judge_acc_bc": 1.0}}})
        merge_metrics(tmp_path, {cls: {"valid": {"judge_lex_f1_macro": 0.6}}})
        out = merge_metrics(tmp_path, {cls: {"valid": {"judge_f1_macro": 0.55}}})
        assert out[cls]["valid"] == {
            "judge_f1_macro": 0.55,  # same key replaced
            "judge_acc_bc": 1.0,  # untouched
            "judge_lex_f1_macro": 0.6,  # other policy survives
        }
