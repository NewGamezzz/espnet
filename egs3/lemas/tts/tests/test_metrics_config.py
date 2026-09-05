from omegaconf import OmegaConf


def test_metrics_config_has_per_language_sets_and_leak_probe():
    cfg = OmegaConf.load("conf/metrics.yaml")
    names = [d["name"] for d in cfg.dataset.test]
    assert names[0] == "lemas_eval_de" and len(names) == 10
    kinds = [m.metric.score_config[0].name for m in cfg.metrics]
    assert "fwhisper_wer" in kinds and "speaker" in kinds
    leak = [m for m in cfg.metrics if m.metric.get("ref_key") == "lang_ref"]
    assert len(leak) == 1


def test_versa_wrapper_importable():
    from src.metrics.versa import VersaMetric

    assert VersaMetric.__name__ == "VersaMetric"
