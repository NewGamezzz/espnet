#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf
from src.system import TTSSystem

from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)
from espnet3.utils.stages_utils import (
    parse_cli_and_stage_args,
    resolve_stages,
    run_stages,
)


def build_parser(stages: Sequence[str]) -> argparse.ArgumentParser:
    """Build the CLI parser for this recipe.

    Inlined from the ASR template so the recipe is self-contained — no
    dependency on `egs3.TEMPLATE.asr`. Only the arguments this recipe
    actually consumes are exposed; add new ones here as needed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        choices=list(stages) + ["all"],
        nargs="+",
        default=["all"],
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument(
        "--training_config",
        default=None,
        type=Path,
        help="Hydra config for training-time stages.",
    )
    parser.add_argument(
        "--inference_config",
        default=None,
        type=Path,
        help="Hydra config for the infer stage.",
    )
    parser.add_argument(
        "--metrics_config",
        default=None,
        type=Path,
        help="Hydra config for the measure stage (metrics).",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=Path,
        help=(
            "Path to a checkpoint to resume `train` from. Sets "
            "training_config.fit.ckpt_path for this invocation only; the "
            "config itself ships `fit: {}` because an unconditional "
            "ckpt_path FileNotFoundErrors on a fresh run (Lightning's "
            "CheckpointConnector treats a literal, nonexistent path as an "
            "error, not a no-op). local/submit_train.sbatch passes this "
            "only when $LAST_CKPT already exists on disk."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be executed without actually running stages.",
    )
    parser.add_argument(
        "--write_requirements",
        action="store_true",
        help="Write requirements.txt alongside each stage log.",
    )
    return parser


DEFAULT_STAGES = [
    "create_dataset",
    "create_shape",
    "train",
    "infer",
    "measure",
]

ALL_STAGES = DEFAULT_STAGES
# NOTE: these name the near-empty *template* stub each recipe config merges
# over (egs3/TEMPLATE/tts/conf/{training,inference,metrics}.yaml -- see
# `load_and_merge_config`'s docstring and the comment in main() below), NOT
# this recipe's own concrete config filenames. `config_name` is passed to
# `load_and_merge_config` unconditionally, independent of whichever real
# file `--training_config`/`--inference_config` points at, so it must always
# match a file that exists under the template package. Found and fixed
# during the final whole-branch review: these previously read
# "training_f5_tts_base.yaml" / "inference_f5_seedtts.yaml" (this recipe's
# own filenames, not the template's), so `load_and_merge_config` raised
# FileNotFoundError trying to open
# egs3/TEMPLATE/tts/conf/training_f5_tts_base.yaml on every single
# invocation -- verified directly; no test exercised run.py's actual CLI
# entrypoint end to end, the same class of gap IMPORTANT 5 closes for
# output_fn and metrics `_target_`. See tests/test_run_default_configs.py.
DEFAULT_TRAINING_CONFIG = "training.yaml"
DEFAULT_INFERENCE_CONFIG = "inference.yaml"
DEFAULT_METRICS_CONFIG = "metrics.yaml"


def main(args) -> None:
    stages_to_run = resolve_stages(args.stages, ALL_STAGES)

    # Each config is merged over the shared `egs3.TEMPLATE.tts` default (kept
    # deliberately near-empty, see egs3/TEMPLATE/tts/conf/*.yaml), not over
    # another recipe config. The recipe ships multiple *independent* full
    # configs (e.g. VITS training.yaml and F5 training_f5_tts.yaml); merging
    # one over another would deep-merge incompatible blocks (e.g.
    # model.tts_conf), and merging a config over itself is a no-op that hides
    # the intended default values entirely. `default_package` is left unset so
    # `load_and_merge_config` auto-infers `egs3.TEMPLATE.tts` from the config
    # path.
    training_config = load_and_merge_config(
        args.training_config,
        config_name=DEFAULT_TRAINING_CONFIG,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        config_name=DEFAULT_INFERENCE_CONFIG,
        resolve=False,
    )
    metrics_config = load_and_merge_config(
        args.metrics_config,
        config_name=DEFAULT_METRICS_CONFIG,
        resolve=False,
    )
    if args.ckpt_path is not None:
        if training_config is None:
            raise ValueError("--ckpt_path requires --training_config.")
        # OmegaConf.update rather than attribute assignment: works whether
        # `fit` is present as `{}` (the shipped default) or absent entirely,
        # and is the same mechanism apply_training_experiment_context below
        # uses to patch config fields in place before resolution.
        OmegaConf.update(
            training_config, "fit.ckpt_path", str(args.ckpt_path), force_add=True
        )
    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=None,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, inference_config)

    system = TTSSystem(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )

    pretrain_stages = {"create_dataset", "create_shape", "train"}
    required_configs = {stage: training_config for stage in pretrain_stages}
    required_configs["infer"] = inference_config
    missing = [
        stage
        for stage in stages_to_run
        if stage in required_configs and required_configs[stage] is None
    ]
    if missing:
        raise ValueError(
            f"Config not provided for stage(s): {', '.join(missing)}. "
            "Use --training_config/--inference_config."
        )

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args)
