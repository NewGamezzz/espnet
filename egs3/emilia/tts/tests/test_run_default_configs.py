"""run.py's own DEFAULT_*_CONFIG constants, and the --ckpt_path override.

Found during the final whole-branch review while verifying IMPORTANT 1:
`DEFAULT_TRAINING_CONFIG`/`DEFAULT_INFERENCE_CONFIG` named this recipe's
OWN config filenames ("training_f5_tts_base.yaml" /
"inference_f5_seedtts.yaml"), but `load_and_merge_config`'s `config_name`
parameter selects the near-empty TEMPLATE stub every recipe config merges
over (`egs3/TEMPLATE/tts/conf/{training,inference,metrics}.yaml`),
independent of whichever concrete file `--training_config` points at.
Verified directly: `python run.py --stages train --training_config
conf/training_f5_tts_base.yaml` raised
``FileNotFoundError: .../egs3/TEMPLATE/tts/conf/training_f5_tts_base.yaml``
before the fix, on every invocation -- no test exercised run.py's actual
CLI entrypoint end to end, the same class of gap IMPORTANT 5 closes for
`output_fn` and metrics `_target_`.
"""

import ast
import runpy
from pathlib import Path

import pytest

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"
TEMPLATE_CONF = (
    Path(__file__).resolve().parents[3] / "TEMPLATE" / "tts" / "conf"
)


def _get_module_constant(name: str) -> str:
    tree = ast.parse(RUN_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found in run.py")


@pytest.mark.parametrize(
    "const_name",
    ["DEFAULT_TRAINING_CONFIG", "DEFAULT_INFERENCE_CONFIG", "DEFAULT_METRICS_CONFIG"],
)
def test_default_config_matches_a_real_template_file(const_name):
    """Each DEFAULT_*_CONFIG must name a file that actually exists under
    egs3/TEMPLATE/tts/conf/ -- that's the only thing config_name is used
    for; it is never compared against this recipe's own filenames."""
    filename = _get_module_constant(const_name)
    assert (TEMPLATE_CONF / filename).is_file(), (
        f"{const_name}={filename!r} does not match any file under "
        f"{TEMPLATE_CONF}; load_and_merge_config will FileNotFoundError."
    )


def test_default_training_config_is_not_this_recipes_own_filename():
    """Regression pin for the exact bug found: config_name must NOT be set
    to this recipe's own conf/ filename."""
    assert (
        _get_module_constant("DEFAULT_TRAINING_CONFIG") != "training_f5_tts_base.yaml"
    )
    assert (
        _get_module_constant("DEFAULT_INFERENCE_CONFIG") != "inference_f5_seedtts.yaml"
    )


def test_all_stages_dry_run_end_to_end(monkeypatch, caplog):
    """The real regression test: invoke run.py's actual __main__ entrypoint
    (not just import main() and hand-build args) with every config, exactly
    the way local/submit_train.sbatch and the README's stage walkthrough do,
    and confirm every stage is reached instead of crashing on the first
    `load_and_merge_config` call."""
    argv = [
        "run.py",
        "--stages",
        "all",
        "--training_config",
        "conf/training_f5_tts_base.yaml",
        "--inference_config",
        "conf/inference_f5_seedtts.yaml",
        "--metrics_config",
        "conf/metrics.yaml",
        "--dry_run",
    ]
    monkeypatch.chdir(RUN_PY.parent)
    monkeypatch.setattr("sys.argv", argv)
    with caplog.at_level("INFO"):
        runpy.run_path(str(RUN_PY), run_name="__main__")
    for stage in ("create_dataset", "create_shape", "train", "infer", "measure"):
        assert f"would run stage: {stage}" in caplog.text


def test_ckpt_path_flag_sets_fit_ckpt_path_only_when_passed():
    """--ckpt_path must patch training_config.fit.ckpt_path for this
    invocation; without it, fit stays exactly what the config ships
    (`{}`, per IMPORTANT 1) so an unconditional resume is never
    reintroduced by accident."""
    import argparse

    import run as run_module
    from omegaconf import OmegaConf

    parser = run_module.build_parser(stages=run_module.DEFAULT_STAGES)
    args = parser.parse_args(
        [
            "--stages",
            "train",
            "--training_config",
            "conf/training_f5_tts_base.yaml",
            "--ckpt_path",
            "/tmp/some.ckpt",
            "--dry_run",
        ]
    )
    assert isinstance(args, argparse.Namespace)
    training_config = run_module.load_and_merge_config(
        args.training_config,
        config_name=run_module.DEFAULT_TRAINING_CONFIG,
        resolve=False,
    )
    assert training_config.fit == {}
    OmegaConf.update(
        training_config, "fit.ckpt_path", str(args.ckpt_path), force_add=True
    )
    assert training_config.fit.ckpt_path == "/tmp/some.ckpt"
