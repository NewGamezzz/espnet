"""Write per-arm inference/metrics configs and 1-GPU sbatch files for the AMI
strata (canonical copy; the Delta copy lives in /work/hdd/bbjs/ttrachu/scripts).

Arms per stratum K in {2,3,4} and per manifest suffix ('' = full stratum,
'_sub20' = pinned 20-window subset): gt, resynth, generate Mode O, generate
Mode T, concat baseline.  One job per arm (Thanapat's rule: independent 1-GPU
short jobs, never a multi-GPU wave), submit no more than four at a time on
/work/hdd (import contention).

Usage:
    python make_ami_runs.py --ckpt /path/backup_step128570.ckpt --tag s128570 [--subset]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# The recipe dir this script lives in: configs, jobs and the sbatch `cd` all
# follow the checkout the generator runs from (a hardcoded path once sent a
# batch of arms into a sibling checkout on another branch).
ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conf" / "generated"
JOBS = ROOT / "jobs"
# The known-good Delta recipe env (hdd pixi; slow import, hence 2 h arms).
PY = (
    "/work/hdd/bbjs/ttrachu/development/espnet3/recipe/f5_tts/"
    "espnet_conversational_f5/tools/.pixi/envs/default/bin/python"
)
# Base-architecture eval training config (model/vocab/feats for the paper
# checkpoint); run.py takes it on the command line, as the ZipVoice arms did.
TRAIN = "conf/generated/training_covomix2_eval.yaml"
ACCOUNT = "bbjs-delta-gpu"

SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time={walltime}
#SBATCH --output={root}/exp/ami/logs/%x_%j.out
set -euo pipefail
cd {root}
mkdir -p exp/ami/logs
export PYTHONUNBUFFERED=1 PYTHONPATH={worktree}:{root}
git log --oneline -1
{py} run.py --stages infer measure --inference_config {inf} --metrics_config {met} \\
    --training_config {train}
"""

MODE_OF = {"gt": "gt", "resynth": "resynth", "O": "generate", "T": "generate",
           "concat": "generate_concat_baseline"}
ARMS = (("gt", "01:00:00"), ("resynth", "01:30:00"), ("O", "02:00:00"),
        ("T", "02:00:00"), ("concat", "02:00:00"))


def _set(text: str, key: str, value: str) -> str:
    """Replace the first ``<indent>key:`` line's value (exact key match)."""
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.M)
    if not pat.search(text):
        raise KeyError(key)
    return pat.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)


def arm(base_inf: str, base_met: str, K: int, suffix: str, arm_name: str,
        tag: str, ckpt: str, walltime: str) -> str:
    mode = MODE_OF[arm_name]
    name = f"ami_k{K}{suffix}_{arm_name}_{tag}"
    inf = _set(base_inf, "mode", mode)
    inf = _set(inf, "test_name", f"ami_k{K}{suffix}")
    inf = _set(inf, "inference_dir", f"${{exp_dir}}/{name}")
    inf = _set(inf, "manifest", f"data/eval/ami_test_k{K}{suffix}_v1.jsonl")
    inf = _set(inf, "num_active_speakers", str(K))
    inf = _set(inf, "text_format", "timestamps" if arm_name == "T" else "order")
    inf = _set(inf, "ckpt", "null" if arm_name in ("gt", "resynth", "concat") else ckpt)
    if arm_name != "O":
        # predicted duration is a generate + order-text policy; every other
        # arm scores the ground-truth length (the rule's estimate is recorded)
        inf = inf.replace("  source: predicted\n", "  source: ground_truth\n", 1)
    if arm_name == "concat":
        inf = inf.replace(f"mode: {mode}\n", f"mode: {mode}\nsource: sssd\n", 1)
    met = _set(base_met, "mode", mode)
    met = _set(met, "inference_dir", f"${{exp_dir}}/{name}")
    met = met.replace("name: ami_k2", f"name: ami_k{K}{suffix}")
    CONF.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    (CONF / f"inference_{name}.yaml").write_text(inf)
    (CONF / f"metrics_{name}.yaml").write_text(met)
    (JOBS / f"run_{name}.sbatch").write_text(
        SBATCH.format(
            name=name, account=ACCOUNT, walltime=walltime, root=ROOT,
            worktree=ROOT.parents[2], py=PY, train=TRAIN,
            inf=CONF / f"inference_{name}.yaml", met=CONF / f"metrics_{name}.yaml",
        )
    )
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", action="store_true")
    a = ap.parse_args(argv)
    suffix = "_sub20" if a.subset else ""
    base_inf = (ROOT / "conf" / "inference_ami.yaml").read_text()
    base_met = (ROOT / "conf" / "metrics_ami.yaml").read_text()
    names = [
        arm(base_inf, base_met, K, suffix, arm_name, a.tag, a.ckpt,
            "02:00:00" if a.subset else wt)  # steps 64 K4 ~130 s/window + measure
        for K in (2, 3, 4)
        for arm_name, wt in ARMS
    ]
    (JOBS / f"submit_ami{suffix}_{a.tag}.sh").write_text(
        "#!/bin/bash\nset -e\n"
        + "".join(f"sbatch {JOBS}/run_{n}.sbatch\nsleep 2\n" for n in names)
    )
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
