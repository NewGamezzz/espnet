"""Submit the subset evaluation arm for every backup checkpoint not yet scored.

``local/eval_watch.sbatch`` runs this every hour on a Delta cpu node (the
arms need Delta's espeak-ng), so a new ``backup_step<N>.ckpt`` of either run
is evaluated without anyone asking. A marker ``exp/<tag>/eval_submitted/
<ckpt>.job`` holds the job id, so each checkpoint is submitted once.
The text-use probe (``local/probe_text_1gpu.sbatch``) runs the same way for
every backup of ``PROBE_RUNS``, marked in ``exp/<tag>/probe_submitted/``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

RUNS: List[Tuple[str, str]] = [
    (
        "train_f5_base_lemas_dualprompt_gh200_lr1.5e-4_r3",
        "conf/training_f5_base_dualprompt_gh200.yaml",
    ),
    ("train_f5_base_lemas_dualprompt_bf16_r3", "conf/training_f5_base_dualprompt.yaml"),
    # paper baseline row D1 (speaker prompt only); its arm configs are below
    ("train_f5_base_lemas_d1_spkonly", "conf/training_f5_base_d1_spkonly.yaml"),
]
# Text-use probe (local/probe_text_use.py) of every backup: the earliest sign
# that a run reads its phones. The layout diagnostics and the stock-layout
# baseline rows (S0c, S0a: no inference path yet) get only the probe.
PROBE_RUNS: List[Tuple[str, str]] = RUNS + [
    ("diag_f5_small_en_dualprompt", "conf/training_f5_small_en_dualprompt.yaml"),
    ("diag_f5_small_en_stock", "conf/training_f5_small_en_stock.yaml"),
    (
        "train_f5_base_lemas_s0c_stock_phones",
        "conf/training_f5_base_s0c_stock_phones.yaml",
    ),
    (
        "train_f5_base_lemas_s0a_stock_chars",
        "conf/training_f5_base_s0a_stock_chars.yaml",
    ),
]
MARKER_DIRS = {"eval": "eval_submitted", "probe": "probe_submitted"}
INFERENCE = "conf/inference_lemas_eval_sub50.yaml"
METRICS = "conf/metrics_sub50.yaml"
# Per-run arm configs where the default two-prompt arm does not apply: D1 never
# saw a language prompt, so it is scored with the speaker-only configs.
ARM_CONFIGS: Dict[str, Tuple[str, str]] = {
    "train_f5_base_lemas_d1_spkonly": (
        "conf/inference_lemas_eval_spk_only_sub50.yaml",
        "conf/metrics_spk_only_sub50.yaml",
    ),
}


@dataclass
class Arm:
    tag: str
    ckpt: str
    train_config: str
    kind: str = "eval"
    inference: str = INFERENCE
    metrics: str = METRICS

    @property
    def command(self) -> str:
        if self.kind == "probe":
            return (
                f"sbatch --parsable -J probe_{self.tag[-12:]}_"
                f"{self.ckpt[len('backup_'):]} local/probe_text_1gpu.sbatch "
                f"{self.train_config} {self.ckpt}"
            )
        return (
            f"LEMAS_CKPT={self.ckpt} sbatch --parsable -J arm_{self.tag[-12:]}_"
            f"{self.ckpt[len('backup_'):]} local/run_arm_1gpu.sbatch "
            f"{self.inference} {self.metrics} {self.train_config}"
        )


def pending_arms(
    recipe_dir, runs: Sequence[Tuple[str, str]], kind: str = "eval"
) -> List[Arm]:
    """Return the ``kind`` jobs to submit, oldest checkpoint first per run."""
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
            if not (exp / MARKER_DIRS[kind] / f"{ckpt.stem}.job").exists():
                inference, metrics = ARM_CONFIGS.get(tag, (INFERENCE, METRICS))
                arms.append(Arm(tag, ckpt.stem, train_config, kind, inference, metrics))
    return arms


def submit_pending(
    recipe_dir, runs, sbatch: Callable[[str], str], kind: str = "eval"
) -> List[Arm]:
    """Submit every pending job through ``sbatch(command) -> job id`` and mark it."""
    done = []
    for arm in pending_arms(recipe_dir, runs, kind):
        job = sbatch(arm.command).strip()
        marker = (
            Path(recipe_dir) / "exp" / arm.tag / MARKER_DIRS[kind] / f"{arm.ckpt}.job"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(job)
        print(f"eval_watch: submitted {kind} {job} for {arm.tag}/{arm.ckpt}")
        done.append(arm)
    return done


def _sbatch(command: str) -> str:
    return subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True
    ).stdout


if __name__ == "__main__":
    arms = submit_pending(Path.cwd(), RUNS, _sbatch)
    arms += submit_pending(Path.cwd(), PROBE_RUNS, _sbatch, kind="probe")
    if not arms:
        print("eval_watch: nothing new")
    sys.exit(0)
