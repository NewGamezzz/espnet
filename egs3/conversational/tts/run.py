#!/usr/bin/env python3
"""Stage runner for the conversational multi-branch F5 recipe.

Stages:
  - ``create_dataset``: SSSD manifests + extended vocab via the step-2
    builder (``dataset/``), driven by ``training_config.dataset`` entries.
  - ``train``: fine-tune the injected multi-branch F5 model
    (``conf/training_poc.yaml``).

Sanity generation is a script, not a stage: ``local/generate_dev.py``.
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

DEFAULT_STAGES = ["create_dataset", "train"]
ALL_STAGES = DEFAULT_STAGES
# Name of the near-empty shared default under egs3/TEMPLATE/tts/conf that
# the recipe config is merged over (see the libritts run.py for why configs
# are merged over the TEMPLATE, not over each other).
DEFAULT_TRAINING_CONFIG = "training.yaml"


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


def main(args) -> None:
    stages_to_run = resolve_stages(args.stages, ALL_STAGES)

    training_config = load_and_merge_config(
        args.training_config,
        config_name=DEFAULT_TRAINING_CONFIG,
        resolve=False,
    )
    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=None,
        metrics_config=None,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, None)

    if training_config is None:
        raise ValueError(
            f"Config not provided for stage(s): {', '.join(stages_to_run)}. "
            "Use --training_config."
        )
    system = ConversationalTTSSystem(training_config=training_config)

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args)
