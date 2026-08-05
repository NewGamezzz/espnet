"""Guards that the shipped recipe configs stay loadable and cluster-agnostic."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[1]

INFERENCE_CONFIGS = [
    "inference_f5.yaml",
    "inference_f5_libritts.yaml",
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


@pytest.mark.parametrize("name", INFERENCE_CONFIGS)
def test_inference_config_train_config_exists(name):
    """Every inference config must point at a training config that is present.

    `--training_config` never overrides an inference config's own
    `model.train_config`, so a stale value here is not caught at the CLI: it
    surfaces as a checkpoint shape mismatch part way into a GPU job. Deleting a
    training config without repointing its referrers has already happened once.
    """
    train_config = _raw(name)["model"]["train_config"]
    assert train_config.startswith("${recipe_dir}/")
    resolved = RECIPE / train_config.removeprefix("${recipe_dir}/")
    assert resolved.is_file(), f"{name} points at missing {train_config}"


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

    # SIM: documented deviation. VERSA cannot load the official UniSpeech
    # wavlm_large_finetune.pth, which is a raw ECAPA_TDNN_SMALL state dict
    # rather than a HuggingFace AutoModelForAudioXVector repo, so the recipe
    # uses the nearest ESPnet-SPK model instead.
    assert by_name["speaker"]["model_tag"] == "espnet/voxcelebs12_ecapa_wavlm_joint"


# Substrings that would tie this recipe to one specific cluster account.
CLUSTER_MARKERS = (
    "/ocean/projects",
    "cis210027p",
    "GPU-shared",
    "GPU-small",
    "v100-32",
    "#SBATCH",
    "/jet/home",
)

# Generated or downloaded trees that are not part of the recipe source.
GENERATED_DIRS = {
    "data",
    "exp",
    "downloads",
    "pretrained",
    "__pycache__",
    "venv",
    "wandb",
    "lightning_logs",
}


def _is_recipe_source(relative: Path) -> bool:
    """Return False for paths that are not part of the recipe source tree.

    Skips generated or downloaded trees, plus dotfiles and dot-directories:
    local-only agent config such as .claude/ is untracked scratch, not recipe
    source, and would otherwise fail these sweeps on a developer's own
    checkout.
    """
    if GENERATED_DIRS.intersection(relative.parts):
        return False
    if any(part.startswith(".") for part in relative.parts):
        return False
    return True


def test_recipe_source_has_no_cluster_specific_content():
    offenders = []
    for path in sorted(RECIPE.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".yaml",
            ".py",
            ".sh",
            ".md",
            ".sbatch",
        }:
            continue
        relative = path.relative_to(RECIPE)
        if not _is_recipe_source(relative):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this file necessarily contains the markers it checks
        text = path.read_text(encoding="utf-8", errors="ignore")
        offenders.extend(
            f"{relative}: {marker}" for marker in CLUSTER_MARKERS if marker in text
        )
    assert offenders == [], "Cluster-specific content found: " + "; ".join(offenders)


def test_submission_scripts_are_gone():
    # Same skip rules as the content scan: a stashed submission script under
    # exp/ is not recipe source and must not fail this sweep.
    stray = [
        path
        for path in RECIPE.rglob("*.sbatch")
        if _is_recipe_source(path.relative_to(RECIPE))
    ]
    assert stray == []
    assert not (RECIPE / "local" / "pooled_wer_from_jsonl.py").exists()


def test_manifest_and_download_helpers_are_kept():
    assert (RECIPE / "local" / "prepare_librispeech_pc.py").is_file()
    assert (RECIPE / "local" / "download_libritts.sh").is_file()
