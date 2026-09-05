"""Per-arm configs and 1-GPU sbatch files for the Fisher long-form arm
(canonical copy; run it on Delta from the checkout).

Arms: ``gt`` (generate_external_gt anchor on the masked full-length ground
truth), ``chorus`` (generate_external_chunked, the stage-2 special-token
recipe of conf/inference_fisher_longform_chunked.yaml), ``concat``
(generate_concat_baseline, stock F5 turn by turn, length-flat by
construction).  ``--ids-file`` pins a subset through
selection.dialogue_ids (the 2-call smoke, the 16 speaker-disjoint calls);
the full set runs as ``--shards N`` independent jobs per arm
(selection.shard_index/shard_count), each a 1-GPU job (Thanapat's rule: no
multi-GPU waves).  Design: "Design - Long-Form Two-Speaker Evaluation on
Fisher" (2026-09-04).

Usage:
    python local/make_fisher_longform_runs.py --ckpt <backup_step14199.ckpt> --tag s14199 \\
        --ids-file conf/generated/fisher_longform_smoke.txt
    python local/make_fisher_longform_runs.py --ckpt <...> --tag s14199 --shards 8
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(
    "/work/hdd/bbjs/ttrachu/development/espnet3/recipe/f5_tts/espnet_ami_eval/"
    "egs3/conversational/tts"
)
# x86 NVMe pixi env for Delta's A100 nodes (full recipe stack imports in ~55 s
# vs 10+ min from the HDD env).  NOT pixi_env/conversational_f5_stage2: that
# one is an aarch64 build for DeltaAI and fails with "Exec format error" here.
PY = "/work/nvme/bbjs/ttrachu/pixi_x86/default/bin/python"
TRAIN_CHORUS = "conf/generated/training_allon_eval.yaml"
TRAIN_CONCAT = "conf/generated/training_covomix2_eval.yaml"
ACCOUNT = "bbjs-delta-gpu"
ARMS = ("gt", "chorus", "concat")

SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time={walltime}
#SBATCH --output={root}/exp/fisher_longform/logs/%x_%j.out
set -euo pipefail
cd {root}
mkdir -p exp/fisher_longform/logs
export PYTHONUNBUFFERED=1 PYTHONPATH={worktree}:{root}
git log --oneline -1
{py} run.py --stages infer measure --inference_config {inf} --metrics_config {met} \\
    --training_config {train}
"""

# The Concat-F5 arm: stock F5 (ckpt null) through the recipe's concatenated
# baseline on the same external manifest, durations mirroring the system's
# rule (predicted + rate prior).  Mirrors the CoVoMix2/LibriTTS concat rows
# except for the duration block and the manifest.
CONCAT_TEMPLATE = """training_config: {train}
exp_tag: fisher_longform
mode: generate_concat_baseline
source: external
device: cuda
inference_dir: ${{exp_dir}}/{out_dir}
test_name: valid
ckpt: null
use_ema: true
prompt_fill: room_tone

testset:
  manifest: downloads/fisher-longform-v1/manifest.jsonl
  name: fisher-longform-v1

duration:
  scale: 1.048
  speed: 1.0
  source: predicted
  rate_prior_chars: 100.0

selection:
  dialogue_ids: {ids}
  min_duration: null
  max_duration: null
  num_dialogues: null
  seed: 0
  shard_index: {shard_index}
  shard_count: {shard_count}

batching:
  max_batch_audio_sec: 240.0
  max_batch_dialogues: 1

sampling:
  steps: 64
  cfg_strength: 3.0
  cfg_sparse_strength: 3.0
  cfg_sparse_max_chars: 0
  sway_sampling_coef: -1.0
  seed: 0
  autocast_dtype: bfloat16

chunk:
  unchunked_max_sec: 100000.0
  turns: null
  target_sec: 25.0
  cross_fade_sec: 0.1
  cond_silence_gate: true
  cond_gate_fill: room_tone
  cond_loudness_norm: true
"""


def _drop_metric(text: str, target_suffix: str) -> str:
    """Remove one ``- metric:`` list entry (the block up to the next ``  - metric:``
    or end of file) whose ``_target_`` ends with ``target_suffix``."""
    blocks = re.split(r"(?m)^(?=  - metric:)", text)
    kept = [b for b in blocks if not (b.startswith("  - metric:") and target_suffix in b)]
    if len(kept) == len(blocks):
        raise KeyError(target_suffix)
    return "".join(kept)


def _set(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.M)
    if not pat.search(text):
        raise KeyError(key)
    return pat.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)


def _names(arm_name: str, tag: str, *, subset: bool, shard_index: int, shard_count: int,
           arm_suffix: str = "") -> tuple[str, str]:
    suffix = "_sub" if subset else (f"_sh{shard_index}of{shard_count}" if shard_count > 1 else "")
    name = f"fisher_longform{suffix}_{arm_name}{arm_suffix}_{tag}"
    out_dir = f"fisher_longform{'_sub' if subset else ''}_{arm_name}{arm_suffix}_{tag}"
    if shard_count > 1:
        out_dir += f"/shard{shard_index}"
    return name, out_dir


def arm(base_inf: str, base_met: str, arm_name: str, tag: str, ckpt: str, *,
        ids_file, shard_index: int, shard_count: int, walltime: str,
        out_conf: Path, out_jobs: Path, root: Path,
        overrides: dict | None = None, arm_suffix: str = "") -> str:
    """``overrides`` (chorus arm only) set UNIQUE leaf keys of the base inference
    yaml by name, e.g. {"source": "ground_truth", "scale": "1.6",
    "cond_prev_sec": "0.0", "cond_prompt_sec": "4.0"}; ``arm_suffix`` names
    the variant (``_gtdur``, ``_sc16``, ...) in every file and directory."""
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm {arm_name!r}; expected one of {ARMS}")
    if overrides and arm_name != "chorus":
        raise ValueError("overrides apply to the chorus arm only")
    subset = ids_file is not None
    name, out_dir = _names(arm_name, tag, subset=subset, shard_index=shard_index,
                           shard_count=shard_count, arm_suffix=arm_suffix)
    ids = str(Path(ids_file).resolve()) if subset else "null"
    if arm_name == "concat":
        inf = CONCAT_TEMPLATE.format(train=TRAIN_CONCAT, out_dir=out_dir, ids=ids,
                                     shard_index=shard_index, shard_count=shard_count)
        mode, train = "generate_concat_baseline", TRAIN_CONCAT
    else:
        mode = "generate_external_gt" if arm_name == "gt" else "generate_external_chunked"
        train = TRAIN_CHORUS
        inf = base_inf
        inf = _set(inf, "mode", mode)
        inf = _set(inf, "inference_dir", f"${{exp_dir}}/{out_dir}")
        inf = _set(inf, "ckpt", "null" if arm_name == "gt" else ckpt)
        inf = _set(inf, "shard_index", str(shard_index))
        inf = _set(inf, "shard_count", str(shard_count))
        inf = _set(inf, "dialogue_ids", ids)
        for key, value in (overrides or {}).items():
            if key in ("seed", "device", "mode", "ckpt", "inference_dir"):
                raise ValueError(f"override of {key!r} is not allowed")
            inf = _set(inf, key, str(value))
    met = _set(base_met, "mode", mode)
    met = _set(met, "inference_dir", f"${{exp_dir}}/{out_dir}")
    if arm_name == "concat":
        # The concatenated baseline writes no aligned gt_wav (its timeline
        # cannot express the reference's gaps and overlap; see
        # src/concat_baseline.py "has_reference_audio"), so the reference-based
        # InteractionMetric cannot score it - the earlier concat rows
        # (CoVoMix2, LibriTTS) were measured without it too.
        met = _drop_metric(met, "interaction.InteractionMetric")
    out_conf.mkdir(parents=True, exist_ok=True)
    out_jobs.mkdir(parents=True, exist_ok=True)
    (out_conf / f"inference_{name}.yaml").write_text(inf)
    (out_conf / f"metrics_{name}.yaml").write_text(met)
    (out_jobs / f"run_{name}.sbatch").write_text(SBATCH.format(
        name=name, account=ACCOUNT, walltime=walltime, root=root, worktree=root.parents[2],
        py=PY, train=train, inf=out_conf / f"inference_{name}.yaml", met=out_conf / f"metrics_{name}.yaml",
    ))
    return name


def generate(*, recipe: Path, out_conf: Path, out_jobs: Path, ckpt: str, tag: str,
             arms=ARMS, ids_file=None, shards: int = 1, walltime: str = "04:00:00",
             root: Path = ROOT, overrides: dict | None = None, arm_suffix: str = "") -> list[str]:
    base_inf = (recipe / "conf" / "inference_fisher_longform_chunked.yaml").read_text()
    base_met = (recipe / "conf" / "metrics_fisher_longform.yaml").read_text()
    names = []
    for arm_name in arms:
        n_shards = 1 if ids_file is not None else shards
        for i in range(n_shards):
            names.append(arm(base_inf, base_met, arm_name, tag, ckpt, ids_file=ids_file,
                             shard_index=i, shard_count=n_shards, walltime=walltime,
                             out_conf=out_conf, out_jobs=out_jobs, root=root,
                             overrides=overrides, arm_suffix=arm_suffix))
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--walltime", default="04:00:00")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="chorus arm only: set a unique leaf key of the base yaml (repeatable)")
    ap.add_argument("--arm-suffix", default="", help="variant name appended to the arm, e.g. _gtdur")
    a = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in a.override)
    names = generate(recipe=ROOT, out_conf=ROOT / "conf" / "generated", out_jobs=ROOT / "jobs",
                     ckpt=a.ckpt, tag=a.tag, arms=a.arms, ids_file=a.ids_file, shards=a.shards,
                     walltime=a.walltime, overrides=overrides, arm_suffix=a.arm_suffix)
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
