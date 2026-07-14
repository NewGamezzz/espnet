#!/usr/bin/env python3
"""Stage runner for the conversational multi-branch F5 recipe.

Stages:
  - ``create_dataset``: SSSD manifests + extended vocab via the step-2
    builder (``dataset/``), driven by ``training_config.dataset`` entries.
  - ``train``: fine-tune the injected multi-branch F5 model
    (``conf/training_poc.yaml``).
  - ``infer``: batch-generate multi-channel conversations (generate / gt /
    resynth) from manifest windows into the measure-stage output contract
    (``src/inference.py``, ``conf/inference_conversational.yaml``).

``local/generate_dev.py`` remains a standalone single-window listening tool.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Imports are package-qualified (egs3.conversational.tts...), so the repo
# root must be importable even when running `python run.py` from here.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from egs3.conversational.tts.src.system import ConversationalTTSSystem  # noqa: E402
from espnet3.utils.config_utils import load_and_merge_config  # noqa: E402
from espnet3.utils.logging_utils import configure_logging  # noqa: E402
from espnet3.utils.run_utils import (  # noqa: E402
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)
from espnet3.utils.stages_utils import (  # noqa: E402
    parse_cli_and_stage_args,
    resolve_stages,
    run_stages,
)

DEFAULT_STAGES = ["create_dataset", "train", "infer"]
ALL_STAGES = DEFAULT_STAGES
# Name of the near-empty shared default under egs3/TEMPLATE/tts/conf that
# the recipe config is merged over (see the libritts run.py for why configs
# are merged over the TEMPLATE, not over each other).
DEFAULT_TRAINING_CONFIG = "training.yaml"
DEFAULT_INFERENCE_CONFIG = "inference.yaml"


def build_parser(stages: Sequence[str]) -> argparse.ArgumentParser:
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
        default=Path("conf/training_poc.yaml"),
        type=Path,
        help="Hydra config for training-time stages.",
    )
    parser.add_argument(
        "--inference_config",
        default=Path("conf/inference_conversational.yaml"),
        type=Path,
        help="Hydra config for the infer stage.",
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


def check_required_configs(stages_to_run, training_config, inference_config) -> None:
    """Fail fast when a requested stage's config is missing.

    The training-time stages need the training config; ``infer`` needs the
    inference config (which self-references its own training config).  Both
    parser defaults point at real recipe configs, so this only fires when a
    flag is explicitly overridden to a missing value.
    """
    training_stages = {"create_dataset", "train"}
    if training_config is None and training_stages.intersection(stages_to_run):
        raise ValueError(
            "Training config not provided for stage(s): "
            f"{', '.join(sorted(training_stages.intersection(stages_to_run)))}. "
            "Use --training_config."
        )
    if "infer" in stages_to_run and inference_config is None:
        raise ValueError(
            "Inference config not provided for the 'infer' stage. "
            "Use --inference_config."
        )


def main(args) -> None:
    stages_to_run = resolve_stages(args.stages, ALL_STAGES)

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
    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        publication_config=None,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, inference_config)

    check_required_configs(stages_to_run, training_config, inference_config)
    system = ConversationalTTSSystem(
        training_config=training_config,
        inference_config=inference_config,
    )

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args)
