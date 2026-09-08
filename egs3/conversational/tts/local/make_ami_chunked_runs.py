"""Write per-arm configs + 1-GPU sbatch files for the AMI windows on the
CHUNKED external path (the ZipVoice-Dialog protocol; Thanapat 2026-09-08).

Input per K: the exported external test set
``exp/ami/external/k<K>[_sub20]_<manifest-tag>/manifest.jsonl``
(local/export_ami_baseline_inputs.py --gt).  Arms:

* ``gt``    - generate_external_gt: the anchor from the exported gt wavs;
* ``sptok`` - special-token prompt, the stage-2 ZipVoice final recipe
              (cfg 3.5, cond_prompt_sec 1.5, cond_prev_sec 5);
* ``trd``   - transcripts wrapper, the all-on ZipVoice tuned recipe
              (cfg 3.0, cond_include_prompt false, cond_history_chunks 1).

One job per (K, arm), walltime from the measured subset table.

Usage:
    python local/make_ami_chunked_runs.py --ckpt <ckpt> --tag s2_50461 --subset \\
        --manifest-tag pool_v2 --training-config conf/generated/training_chorus_eval.yaml --arms gt sptok
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conf" / "generated"
JOBS = ROOT / "jobs"
PY = (
    "/work/hdd/bbjs/ttrachu/development/espnet3/recipe/f5_tts/"
    "espnet_conversational_f5/tools/.pixi/envs/default/bin/python"
)
ACCOUNT = "bbjs-delta-gpu"
SUBSET_WALLTIME = {"gt": "01:20:00", 2: "01:45:00", 3: "02:15:00", 4: "02:30:00"}
FULL_WALLTIME = {"gt": "02:00:00", 2: "04:00:00", 3: "04:00:00", 4: "04:00:00"}

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
    """Replace the first ``<indent>key:`` line's value (exact key match)."""
    pat = re.compile(rf"^(\s*{re.escape(key)}:)[^\n]*$", re.M)
    if not pat.search(text):
        raise KeyError(key)
    return pat.sub(lambda m: f"{m.group(1)} {value}", text, count=1)


def arm(base_inf: str, base_met: str, K: int, suffix: str, arm_name: str, tag: str,
        ckpt: str, train: str, manifest_tag: str, walltime: str) -> str:
    testset = f"k{K}{suffix}_{manifest_tag}"
    name = f"ami_{testset}_ch_{arm_name}_{tag}"
    out_dir = name
    inf = base_inf
    inf = _set(inf, "training_config", train)
    inf = _set(inf, "mode", "generate_external_gt" if arm_name == "gt" else "generate_external_chunked")
    inf = _set(inf, "inference_dir", f"${{exp_dir}}/{out_dir}")
    inf = _set(inf, "ckpt", "null" if arm_name == "gt" else ckpt)
    inf = _set(inf, "manifest", f"exp/ami/external/{testset}/manifest.jsonl")
    inf = _set(inf, "name", f"ami_{testset}")
    if arm_name == "trd":
        # the all-on transcripts recipe: the wrapper knobs replace the sptok
        # ones line by line (never a literal block match, which silently
        # leaves the sptok format in place when a value changes).
        inf = _set(inf, "cond_format", "transcripts")
        inf = re.sub(r"^\s*cond_prompt_sec:[^\n]*\n", "  cond_include_prompt: false\n", inf, count=1, flags=re.M)
        inf = re.sub(r"^\s*cond_prev_sec:[^\n]*\n", "  cond_history_chunks: 1\n", inf, count=1, flags=re.M)
        inf = _set(inf, "cfg_strength", "3.0")
        if "cond_format: special_tokens" in inf or "cond_prompt_sec" in inf:
            raise RuntimeError("trd arm still carries special-token knobs")
    met = _set(base_met, "mode", "generate_external_gt" if arm_name == "gt" else "generate_external_chunked")
    met = _set(met, "inference_dir", f"${{exp_dir}}/{out_dir}")
    (CONF / f"inference_{name}.yaml").write_text(inf)
    (CONF / f"metrics_{name}.yaml").write_text(met)
    (JOBS / f"run_{name}.sbatch").write_text(SBATCH.format(
        name=name, account=ACCOUNT, walltime=walltime, root=ROOT, worktree=ROOT.parents[2],
        py=PY, inf=CONF / f"inference_{name}.yaml", met=CONF / f"metrics_{name}.yaml", train=train,
    ))
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--manifest-tag", default="pool_v2")
    ap.add_argument("--training-config", default="conf/generated/training_chorus_eval.yaml")
    ap.add_argument("--arms", nargs="*", default=["gt", "sptok", "trd"])
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 3, 4])
    a = ap.parse_args(argv)
    CONF.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    suffix = "_sub20" if a.subset else ""
    base_inf = (ROOT / "conf" / "inference_ami_windows_chunked.yaml").read_text()
    base_met = (ROOT / "conf" / "metrics_ami_windows_chunked.yaml").read_text()
    table = SUBSET_WALLTIME if a.subset else FULL_WALLTIME
    for K in a.ks:
        for arm_name in a.arms:
            wall = table["gt"] if arm_name == "gt" else table[K]
            print(arm(base_inf, base_met, K, suffix, arm_name, a.tag, a.ckpt, a.training_config,
                      a.manifest_tag, wall))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
