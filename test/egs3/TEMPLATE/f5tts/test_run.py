"""Tests for the egs3/TEMPLATE/f5tts defaults and runner wiring."""

from argparse import Namespace
from pathlib import Path

import pytest
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from egs3.TEMPLATE.f5tts.run import DEFAULT_STAGES, build_parser, main
from espnet3.systems.f5tts.system import F5TTSSystem
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.f5tts"


def _run_arguments(**overrides):
    arguments = {
        "stages": ["all"],
        "training_config": None,
        "inference_config": None,
        "metrics_config": None,
        "publication_config": None,
        "demo_config": None,
        "dry_run": False,
        "write_requirements": False,
    }
    arguments.update(overrides)
    return Namespace(**arguments)


def test_default_stages_run_manifest_consumers_after_create_dataset() -> None:
    """Execution follows list order, whatever order ``--stages`` is given in."""
    assert DEFAULT_STAGES == [
        "create_dataset",
        "remove_long_short",
        "create_token_list",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]


def test_every_default_stage_is_a_system_method() -> None:
    for stage in DEFAULT_STAGES:
        assert callable(getattr(F5TTSSystem, stage))


def test_parser_accepts_the_stage_names_and_config_options() -> None:
    parser = build_parser(stages=DEFAULT_STAGES)
    args = parser.parse_args(
        [
            "--stages",
            "remove_long_short",
            "create_token_list",
            "--training_config",
            "conf/training.yaml",
            "--demo_config",
            "conf/demo.yaml",
        ]
    )

    assert args.stages == ["remove_long_short", "create_token_list"]
    assert args.training_config == Path("conf/training.yaml")
    assert args.demo_config == Path("conf/demo.yaml")
    assert args.publication_config is None


def test_training_defaults_build_an_espnet3_native_f5tts() -> None:
    config = load_default_config("training.yaml", PACKAGE)

    # No ESPnet2 task bridge: the model is instantiated from its target.
    assert config.task is None
    assert config.model._target_ == "espnet3.systems.f5tts.f5tts.F5TTS"
    assert get_class(config.model._target_).__name__ == "F5TTS"
    assert get_class(config.scheduler._target_).__name__ == "LinearWarmupDecayLR"
    assert get_class(config.trainer.callbacks[0]._target_).__name__ == "EMACallback"
    assert (
        config.dataset._target_
        == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert config.dataset._recursive_ is False
    # DataOrganizer sets the preprocessor's train flag per split.
    assert "train" not in config.dataset.preprocessor
    assert config.create_dataset.recipe_dir == "."


def test_training_defaults_share_one_token_list_and_one_mel_setup() -> None:
    config = load_default_config("training.yaml", PACKAGE)
    OmegaConf.resolve(config)

    assert config.token_list == "./data/token_list/tokens.txt"
    assert config.model.token_list == config.token_list
    assert config.dataset.preprocessor.token_list == config.token_list
    assert config.dataset.preprocessor.token_type == config.create_token_list.token_type
    assert OmegaConf.to_container(config.model.feats_extract_config) == {
        "fs": 24000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 100,
    }
    # The token list is built from the manifest remove_long_short writes.
    assert config.create_token_list.manifest_path == (
        "./data/manifest_filtered/train.tsv"
    )
    # The schedule ends where training does.
    assert config.scheduler.total_steps == config.trainer.max_steps
    assert config.scheduler.total_steps > config.scheduler.warmup_steps


def test_inference_defaults_name_the_inference_contract_class() -> None:
    config = load_default_config("inference.yaml", PACKAGE)

    assert config.model._target_ == "espnet3.systems.f5tts.inference.Inference"
    inference_class = get_class(config.model._target_)
    assert list(config.input_key) == [field.name for field in inference_class.inputs]
    assert config.batch_size is None
    assert config.output_fn == "src.inference.build_output"
    assert config.output_artifacts.wav.type == "wav"
    assert config.dataset._recursive_ is False
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        config.runner._target_
        == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_template_output_fn_is_importable() -> None:
    """The ``output_fn`` default names a function this template ships."""
    build_output = get_method("egs3.TEMPLATE.f5tts.src.inference.build_output")
    assert callable(build_output)


def test_metrics_defaults_enable_no_metric() -> None:
    config = load_default_config("metrics.yaml", PACKAGE)

    # An empty list, not null: the measure stage iterates over it.
    assert list(config.metrics) == []


def test_publication_defaults_bundle_what_the_model_is_rebuilt_from() -> None:
    config = load_default_config("publication.yaml", PACKAGE)

    include = list(config.pack_model.include)
    assert "conf" in include
    assert "src" in include
    assert any(str(path).endswith("token_list") for path in include)
    # last.ckpt is what the packed inference config loads, so it stays.
    assert "last.ckpt" not in config.pack_model.exclude
    assert "**/step*.ckpt" in config.pack_model.exclude
    assert config.pack_model.readme.endswith(
        "egs3/TEMPLATE/f5tts/src/hf_model_readme.md"
    )
    assert config.upload_model["update"] is False


def test_demo_defaults_wire_text_and_reference_to_audio() -> None:
    config = load_default_config("demo.yaml", PACKAGE)

    assert [(spec.key, spec.type) for spec in config.ui.inputs] == [
        ("text", "text"),
        ("reference_speech", "audio"),
        ("reference_text", "text"),
    ]
    assert [(spec.key, spec.type) for spec in config.ui.outputs] == [("wav", "audio")]
    assert config.ui.app_script == "src/app.py"
    assert config.model.trust_user_code is True
    assert config.pack.readme.endswith("egs3/TEMPLATE/f5tts/src/hf_demo_readme.md")
    assert config.upload_demo["update"] is False


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path) -> None:
    user = tmp_path / "training_small.yaml"
    user.write_text(
        "model:\n  hidden_size: 768\n  depth: 18\ntrainer:\n  max_steps: 600000\n",
        encoding="utf-8",
    )

    config = load_and_merge_config(user, "training.yaml", default_package=PACKAGE)

    assert config.exp_tag == "training_small"
    assert config.model.hidden_size == 768
    assert config.model._target_ == "espnet3.systems.f5tts.f5tts.F5TTS"
    assert config.scheduler.total_steps == 600000
    assert config.remove_long_short.min_wav_duration == 1.0


def test_load_and_merge_config_none_path_returns_none() -> None:
    assert load_and_merge_config(None, "demo.yaml", default_package=PACKAGE) is None


@pytest.mark.parametrize(
    "stage",
    ["remove_long_short", "create_token_list", "train", "infer", "pack_model"],
)
def test_main_refuses_a_stage_without_its_config(stage) -> None:
    with pytest.raises(ValueError, match="Config not provided"):
        main(args=_run_arguments(stages=[stage]), system_cls=F5TTSSystem)


def test_main_runs_the_data_stages(recipe_dir) -> None:
    """``run.py`` dispatches the two F5-TTS data stages on a real recipe."""
    import numpy as np
    import soundfile as sf

    manifest_dir = recipe_dir / "data" / "manifest"
    manifest_dir.mkdir()
    for split in ("train", "valid"):
        rows = []
        for name, seconds in (("short", 0.5), ("mid", 2.0)):
            wav_path = recipe_dir / "data" / f"{split}_{name}.wav"
            sf.write(wav_path, np.zeros(int(seconds * 24000), dtype=np.float32), 24000)
            rows.append(f"{split}_{name}\t{wav_path}\tab cab\tspk\n")
        (manifest_dir / f"{split}.tsv").write_text("".join(rows), encoding="utf-8")
    (recipe_dir / "data" / "token_list" / "tokens.txt").unlink()

    main(
        args=_run_arguments(
            # Given out of order on purpose: execution follows DEFAULT_STAGES.
            stages=["create_token_list", "remove_long_short"],
            training_config=Path("conf/training.yaml"),
        ),
        system_cls=F5TTSSystem,
    )

    filtered = (recipe_dir / "data" / "manifest_filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == ["train_mid"]
    tokens = (recipe_dir / "data" / "token_list" / "tokens.txt").read_text()
    assert tokens.splitlines() == [
        "<blank>",
        "<unk>",
        "a",
        "b",
        "<space>",
        "c",
        "<sos/eos>",
    ]
