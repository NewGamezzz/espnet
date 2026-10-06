"""Submit the subset evaluation arm for every backup checkpoint not yet scored.

``local/eval_watch.sbatch`` runs this every hour on a Delta cpu node (the
arms need Delta's espeak-ng), so a new ``backup_step<N>.ckpt`` of either run
is evaluated without anyone asking. A marker ``exp/<tag>/eval_submitted/
<ckpt>.job`` holds the job id, so each checkpoint is submitted once.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

RUNS: List[Tuple[str, str]] = [
    ("train_f5_base_lemas_dualprompt_r2", "conf/training_f5_base_dualprompt.yaml"),
    (
        "train_f5_base_lemas_dualprompt_gh200_lr1.5e-4",
        "conf/training_f5_base_dualprompt_gh200.yaml",
    ),
]
INFERENCE = "conf/inference_lemas_eval_sub50.yaml"
METRICS = "conf/metrics_sub50.yaml"


@dataclass
class Arm:
    tag: str
    ckpt: str
    train_config: str

    @property
    def command(self) -> str:
        return (
            f"LEMAS_CKPT={self.ckpt} sbatch --parsable -J arm_{self.tag[-12:]}_"
            f"{self.ckpt[len('backup_'):]} local/run_arm_1gpu.sbatch "
            f"{INFERENCE} {METRICS} {self.train_config}"
        )


def pending_arms(recipe_dir, runs: Sequence[Tuple[str, str]]) -> List[Arm]:
    """Return the arms to submit, oldest checkpoint first per run."""
    arms = []
    for tag, train_config in runs:
        exp = Path(recipe_dir) / "exp" / tag
        backups = [
            p
            for p in exp.glob("backup_step*.ckpt")
            if p.stem[
                len("backup_step") :
            ].isdigit()  # not the -v1 of a re-fired epoch end
        ]
        for ckpt in sorted(backups, key=lambda p: int(p.stem[len("backup_step") :])):
            if not (exp / "eval_submitted" / f"{ckpt.stem}.job").exists():
                arms.append(Arm(tag, ckpt.stem, train_config))
    return arms


def submit_pending(recipe_dir, runs, sbatch: Callable[[str], str]) -> List[Arm]:
    """Submit every pending arm through ``sbatch(command) -> job id`` and mark it."""
    done = []
    for arm in pending_arms(recipe_dir, runs):
        job = sbatch(arm.command).strip()
        marker = (
            Path(recipe_dir) / "exp" / arm.tag / "eval_submitted" / f"{arm.ckpt}.job"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(job)
        print(f"eval_watch: submitted {job} for {arm.tag}/{arm.ckpt}")
        done.append(arm)
    return done


def _sbatch(command: str) -> str:
    return subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True
    ).stdout


if __name__ == "__main__":
    arms = submit_pending(Path.cwd(), RUNS, _sbatch)
    if not arms:
        print("eval_watch: nothing new")
    sys.exit(0)
