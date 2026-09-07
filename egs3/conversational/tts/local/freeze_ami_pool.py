"""Freeze the six AMI pool-prompt manifests (K=2/3/4 x full/sub20) in ONE
process: each make_eval_manifest CLI call re-imports torch (10-15 min cold on
/work/hdd), which is what timed out the six-call sbatch loop at 1 h.

Usage (from the recipe dir, PYTHONPATH set):
    python local/freeze_ami_pool.py pool_v1
"""
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from egs3.conversational.tts.src.eval_manifest import (  # noqa: E402
    build_eval_manifest,
    write_eval_manifest,
)


def window_ids(path: Path) -> list[str]:
    return [json.loads(l)["window_id"] for l in path.read_text().splitlines()[1:] if l.strip()]


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "pool_v1"
    base = OmegaConf.load("conf/inference_ami.yaml")
    train = OmegaConf.load("conf/generated/training_covomix2_eval.yaml")
    for K in (2, 3, 4):
        for suffix, sel in (
            ("", dict(per_session_cap=12, num_windows=None)),
            ("_sub20", dict(per_session_cap=1, num_windows=20, seed=1)),
        ):
            cfg = OmegaConf.merge(base, OmegaConf.create({"selection": dict(num_active_speakers=K, **sel)}))
            t = time.time()
            header, rows = build_eval_manifest(cfg, training_config=train)
            out = Path(f"data/eval/ami_test_k{K}{suffix}_{tag}.jsonl")
            write_eval_manifest(out, header, rows)
            ref = Path(f"data/eval/ami_test_k{K}{suffix}_v1.jsonl")
            same = window_ids(ref) == window_ids(out) if ref.exists() else None
            pool = header["prompt_pool"]
            print(
                f"K={K}{suffix or ' full'}: {len(rows)} windows, skipped {header['num_skipped']}, "
                f"pool {pool['num_speakers']} spk / {pool['num_items']} utts, "
                f"windows identical to v1: {same}, {time.time() - t:.0f}s -> {out}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
