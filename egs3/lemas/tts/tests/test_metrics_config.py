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


def test_sub50_configs_are_the_full_ones_on_the_first_50_rows_per_language():
    # Thanapat's rule: diagnostic arms on a pinned subset, full runs for final numbers
    for name in ("inference_lemas_eval", "inference_lemas_eval_spk_only"):
        full = OmegaConf.to_container(OmegaConf.load(f"conf/{name}.yaml"))
        sub = OmegaConf.to_container(OmegaConf.load(f"conf/{name}_sub50.yaml"))
        for a, b in zip(full["dataset"]["test"], sub["dataset"]["test"]):
            assert b["name"] == a["name"].replace("lemas_eval", "lemas_eval_sub50")
            assert b["data_src_args"].pop("max_rows") == 50
            b["name"] = a["name"]
        assert sub == full
    met = OmegaConf.to_container(OmegaConf.load("conf/metrics.yaml"))
    sub = OmegaConf.to_container(OmegaConf.load("conf/metrics_sub50.yaml"))
    assert [d["name"] for d in sub["dataset"]["test"]] == [
        d["name"].replace("lemas_eval", "lemas_eval_sub50")
        for d in met["dataset"]["test"]
    ]
    assert sub["metrics"] == met["metrics"]


def test_inference_configs_take_the_checkpoint_from_the_environment(monkeypatch):
    # last.ckpt moves while a chain runs; an arm scores a frozen backup and
    # writes under a directory named after it, so checkpoints never collide
    from espnet3.utils.config_utils import load_config_with_defaults

    monkeypatch.setenv("LEMAS_CKPT", "backup_step75000")
    for name in ("inference_lemas_eval", "inference_lemas_eval_sub50"):
        cfg = load_config_with_defaults(f"conf/{name}.yaml", resolve=False)
        cfg.exp_tag = "tag"
        OmegaConf.resolve(cfg)
        assert cfg.model.checkpoint_path == "./exp/tag/backup_step75000.ckpt"
        assert cfg.inference_dir == f"./exp/tag/{name}_backup_step75000"
    monkeypatch.delenv("LEMAS_CKPT")
    cfg = load_config_with_defaults("conf/inference_lemas_eval.yaml", resolve=False)
    cfg.exp_tag = "tag"
    OmegaConf.resolve(cfg)
    assert cfg.model.checkpoint_path == "./exp/tag/last.ckpt"
    assert cfg.inference_dir == "./exp/tag/inference_lemas_eval_last"


def test_spk_only_metrics_configs_name_the_spk_only_test_sets():
    # arm B writes lemas_eval_spk_<lang>; the metrics must look there
    for suffix in ("", "_sub50"):
        met = OmegaConf.to_container(OmegaConf.load(f"conf/metrics{suffix}.yaml"))
        spk = OmegaConf.to_container(
            OmegaConf.load(f"conf/metrics_spk_only{suffix}.yaml")
        )
        inf = OmegaConf.load(f"conf/inference_lemas_eval_spk_only{suffix}.yaml")
        assert [d["name"] for d in spk["dataset"]["test"]] == [
            d.name for d in inf.dataset.test
        ]
        assert spk["metrics"] == met["metrics"]
