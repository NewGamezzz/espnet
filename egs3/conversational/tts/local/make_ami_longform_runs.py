"""Per-arm configs and 1-GPU sbatch files for the AMI long-form arm
(canonical copy; run it on Delta from the checkout).

Arms: ``gt`` (generate_external_gt anchor, masked full-length ground truth),
``cover`` (chunked, chunk.cover_all_speakers), ``reanchor`` (chunked,
cond_include_prompt instead of coverage).  ``--subset`` pins the six
a-sessions through selection.dialogue_ids; the full set runs as
``--shards N`` independent jobs per arm (selection.shard_index/shard_count),
each a 1-GPU job (Thanapat's rule: no multi-GPU waves).

Usage:
    python local/make_ami_longform_runs.py --ckpt <backup_step128570.ckpt> --tag s128570 --subset
    python local/make_ami_longform_runs.py --ckpt <...> --tag s128570 --shards 4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(
    "/work/hdd/bbjs/ttrachu/development/espnet3/recipe/f5_tts/espnet_ami_eval/"
    "egs3/conversational/tts"
)
CONF = ROOT / "conf" / "generated"
JOBS = ROOT / "jobs"
PY = (
    "/work/hdd/bbjs/ttrachu/development/espnet3/recipe/f5_tts/"
    "espnet_conversational_f5/tools/.pixi/envs/default/bin/python"
)
TRAIN = "conf/generated/training_covomix2_eval.yaml"
ACCOUNT = "bbjs-delta-gpu"
A_SESSIONS = ("EN2002a", "ES2004a", "ES2014a", "IS1009a", "TS3003a", "TS3007a")

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


def _set(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.M)
    if not pat.search(text):
        raise KeyError(key)
    return pat.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)


def arm(base_inf: str, base_met: str, arm_name: str, tag: str, ckpt: str, *,
        subset: bool, shard_index: int, shard_count: int, walltime: str) -> str:
    suffix = "_asess" if subset else (f"_sh{shard_index}of{shard_count}" if shard_count > 1 else "")
    name = f"ami_longform{suffix}_{arm_name}_{tag}"
    out_dir = f"ami_longform{'_asess' if subset else ''}_{arm_name}_{tag}"
    if shard_count > 1:
        out_dir += f"/shard{shard_index}"
    inf = base_inf
    inf = _set(inf, "mode", "generate_external_gt" if arm_name == "gt" else "generate_external_chunked")
    inf = _set(inf, "inference_dir", f"${{exp_dir}}/{out_dir}")
    inf = _set(inf, "ckpt", "null" if arm_name == "gt" else ckpt)
    inf = _set(inf, "shard_index", str(shard_index))
    inf = _set(inf, "shard_count", str(shard_count))
    if subset:
        ids = CONF / "ami_longform_asessions.txt"
        ids.parent.mkdir(parents=True, exist_ok=True)
        ids.write_text("".join(f"{m}\n" for m in A_SESSIONS))
        inf = _set(inf, "dialogue_ids", str(ids))
    if arm_name == "reanchor":
        inf = _set(inf, "cover_all_speakers", "false")
        inf = inf.replace("  cross_fade_sec:", "  cond_include_prompt: true\n  cross_fade_sec:", 1)
    met = _set(base_met, "mode", "generate_external_gt" if arm_name == "gt" else "generate_external_chunked")
    met = _set(met, "inference_dir", f"${{exp_dir}}/{out_dir}")
    CONF.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    (CONF / f"inference_{name}.yaml").write_text(inf)
    (CONF / f"metrics_{name}.yaml").write_text(met)
    (JOBS / f"run_{name}.sbatch").write_text(SBATCH.format(
        name=name, account=ACCOUNT, walltime=walltime, root=ROOT, worktree=ROOT.parents[2],
        py=PY, train=TRAIN, inf=CONF / f"inference_{name}.yaml", met=CONF / f"metrics_{name}.yaml",
    ))
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--arms", nargs="*", default=["gt", "cover", "reanchor"])
    a = ap.parse_args(argv)
    base_inf = (ROOT / "conf" / "inference_ami_longform_chunked.yaml").read_text()
    base_met = (ROOT / "conf" / "metrics_ami_longform.yaml").read_text()
    names = []
    for arm_name in a.arms:
        wall = "02:00:00" if arm_name == "gt" else "04:00:00"
        shards = 1 if a.subset else a.shards
        for i in range(shards):
            names.append(arm(base_inf, base_met, arm_name, a.tag, a.ckpt, subset=a.subset,
                             shard_index=i, shard_count=shards, walltime=wall))
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
