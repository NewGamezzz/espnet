"""Metric measurement entrypoint for hypothesis/reference outputs."""

import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.measurements.abs_measurement import AbsMeasurements
from espnet3.utils.scp_utils import get_cls_path, load_scp_fields


def measure(measures_config: DictConfig):
    """Compute metrics for each test set and write a measures JSON file.

    Args:
        config: Hydra/omegaconf configuration with inference and metric settings.

    Returns:
        Nested dict keyed by metric class path and test set name.
    """
    test_sets = [t.name for t in measures_config.dataset.test]
    results = {}
    assert hasattr(measures_config, "measurements"), "Please specify `measurements`!"

    for measure_config in measures_config.measurements:
        measure = instantiate(measure_config.measure)
        if not isinstance(measure, AbsMeasurements):
            raise TypeError(f"{type(measure)} is not a valid AbsMeasurements instance")

        results[get_cls_path(measure)] = {}
        for test_name in test_sets:
            if hasattr(measure_config, "inputs"):
                inputs = OmegaConf.to_container(measure_config.inputs, resolve=True)
            else:
                ref_key = getattr(measure, "ref_key", None)
                hyp_key = getattr(measure, "hyp_key", None)
                if ref_key is None or hyp_key is None:
                    raise ValueError(
                        f"Metric {get_cls_path(measure)} requires inputs in config"
                    )
                inputs = [ref_key, hyp_key]
            data = load_scp_fields(
                inference_dir=Path(measures_config.inference_dir),
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            measure_result = measure(data, test_name, measures_config.inference_dir)
            results[get_cls_path(measure)].update({test_name: measure_result})

    out_path = Path(measures_config.inference_dir) / "measurements.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
