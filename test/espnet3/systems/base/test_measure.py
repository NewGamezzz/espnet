import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.components.measurements.abs_measurement import AbsMeasurements
from espnet3.systems.base.measurement import measure
from espnet3.utils.scp_utils import get_cls_path


class DummyMetric(AbsMeasurements):
    ref_key = "ref"
    hyp_key = "hyp"

    def __call__(self, data, test_name, inference_dir):
        return {"count": len(data["utt_id"])}


class NoKeyMetric(AbsMeasurements):
    def __call__(self, data, test_name, inference_dir):
        return {"ok": True}


class NotMetric:
    pass


def _write_scp(path: Path, entries):
    path.write_text("\n".join(entries), encoding="utf-8")


def test_measure_uses_metric_keys_and_writes_json(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 r1", "utt2 r2"])
    _write_scp(task_dir / "hyp.scp", ["utt1 h1", "utt2 h2"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    results = measure(cfg)

    expected_key = get_cls_path(DummyMetric())
    assert results[expected_key][test_name] == {"count": 2}
    measures_path = inference_dir / "measurements.json"
    assert measures_path.is_file()
    assert json.loads(measures_path.read_text(encoding="utf-8")) == results


def test_measure_uses_config_inputs_mapping(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "text.scp", ["utt1 r1"])
    _write_scp(task_dir / "hypothesis.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.NoKeyMetric"},
                    "inputs": {"ref": "text", "hyp": "hypothesis"},
                }
            ],
        }
    )

    results = measure(cfg)

    expected_key = get_cls_path(NoKeyMetric())
    assert results[expected_key][test_name] == {"ok": True}


def test_measure_rejects_non_metric_instance(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.NotMetric"}}],
        }
    )

    with pytest.raises(TypeError, match="not a valid AbsMeasurements instance"):
        measure(cfg)


def test_measure_requires_inputs_when_metric_has_no_keys(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.NoKeyMetric"}}],
        }
    )

    with pytest.raises(ValueError, match="requires inputs in config"):
        measure(cfg)
