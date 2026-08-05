"""Guards that the shipped recipe configs stay loadable and cluster-agnostic."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[1]

INFERENCE_CONFIGS = [
    "inference.yaml",
    "inference_f5.yaml",
    "inference_f5_libritts.yaml",
    "inference_pretrained_f5.yaml",
]


def _load(monkeypatch, name, config_name):
    """Load a recipe config the way run.py does, from the recipe directory."""
    monkeypatch.chdir(RECIPE)
    return load_and_merge_config(
        Path("conf") / name, config_name=config_name, resolve=False
    )


def _raw(name):
    """Return the config as plain dicts with interpolations left unresolved.

    OmegaConf resolves interpolations lazily on attribute access, so reading
    `cfg.model.train_config` off a loaded config would hand back the resolved
    path rather than the literal `${recipe_dir}/...` string these tests assert
    on. `to_container(resolve=False)` is what actually preserves them.
    """
    return OmegaConf.to_container(OmegaConf.load(RECIPE / "conf" / name), resolve=False)


@pytest.mark.parametrize("name", INFERENCE_CONFIGS)
def test_inference_configs_load(monkeypatch, name):
    cfg = _load(monkeypatch, name, "inference.yaml")
    assert "dataset" in cfg
    assert "model" in cfg


def test_default_f5_config_uses_librispeech_pc():
    test_sets = _raw("inference_f5.yaml")["dataset"]["test"]
    assert len(test_sets) == 1
    assert test_sets[0]["name"] == "librispeech_pc"
    assert test_sets[0]["data_src"] == "egs3.libritts.tts.dataset.librispeech_pc"
    assert test_sets[0]["data_src_args"]["fs"] == 24000


def test_default_f5_config_is_portable():
    cfg = _raw("inference_f5.yaml")
    assert (
        cfg["model"]["train_config"] == "${recipe_dir}/conf/training_f5_tts_small.yaml"
    )
    assert cfg["model"]["ckpt_path"] == "${exp_dir}/last.ckpt"
    # Empty exp_tag means this config is training-backed: run.py must be given
    # --training_config alongside it (espnet3/utils/run_utils.py's
    # validate_experiment_context returns early when training_config is set).
    assert not cfg["exp_tag"]


def test_libritts_config_keeps_cross_speaker_protocol():
    test_sets = _raw("inference_f5_libritts.yaml")["dataset"]["test"]
    assert [entry["name"] for entry in test_sets] == ["valid", "test"]
    for entry in test_sets:
        assert entry["data_src_args"]["ref_mode"] == "cross_speaker"


def test_librispeech_pc_side_config_is_gone():
    assert not (RECIPE / "conf" / "inference_f5_librispeech_pc.yaml").exists()
    assert not (
        RECIPE / "conf" / "inference_pretrained_f5_librispeech_pc.yaml"
    ).exists()


def test_metrics_config_matches_official_protocol():
    cfg = _raw("metrics.yaml")
    assert [entry["name"] for entry in cfg["dataset"]["test"]] == ["librispeech_pc"]

    score_config = cfg["metrics"][0]["metric"]["score_config"]
    by_name = {entry["name"]: entry for entry in score_config}

    # WER: faster-whisper large-v3, beam 5, float16 - engine-identical to the
    # official eval_librispeech_test_clean.py.
    wer = by_name["fwhisper_wer"]
    assert wer["model_tag"] == "large-v3"
    assert wer["beam_size"] == 5
    assert wer["compute_type"] == "float16"
    assert wer["text_cleaner"] == "whisper_basic"

    # UTMOS only: dnsmos is not part of the official protocol.
    assert by_name["pseudo_mos"]["predictor_types"] == ["utmos"]

    # SIM: documented deviation, VERSA can only load ESPnet-SPK checkpoints.
    assert by_name["speaker"]["model_tag"] == "espnet/voxcelebs12_ecapa_wavlm_joint"
