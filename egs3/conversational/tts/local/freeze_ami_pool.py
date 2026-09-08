"""Freeze the AMI evaluation manifests (K=2/3/4 x full/sub20) in ONE process:
each make_eval_manifest CLI call re-imports torch (10-15 min cold on
/work/hdd), which is what timed out a six-call sbatch loop at 1 h.

Two prompt sources, two tag families:
* ``<tag>`` with ``--pool`` (default): LibriTTS pool prompts (prompt.pool from
  conf/inference_ami.yaml), the PRIMARY protocol;
* ``<tag>`` with ``--corpus``: AMI headset prompts (prompt.pool=null, the
  ladder + gate), the secondary row.

Full strata carry NO per-meeting cap (Thanapat 2026-09-08: cover the whole
partition); the sub20 subsets keep cap 1 / 20 windows / seed 1.

Usage (from the recipe dir, PYTHONPATH set):
    python local/freeze_ami_pool.py pool_v2 --pool
    python local/freeze_ami_pool.py v2 --corpus
"""
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--corpus", action="store_true", help="AMI headset prompts (prompt.pool=null)")
    ap.add_argument("--compare", default=None, help="tag whose window ids to compare against")
    ap.add_argument("--pool-band", type=float, nargs=2, default=None,
                    help="override prompt.pool.band, e.g. 4.0 5.5 for 4 s speaker prompts")
    a = ap.parse_args()
    base = OmegaConf.load("conf/inference_ami.yaml")
    if a.pool_band is not None:
        base = OmegaConf.merge(base, OmegaConf.create({"prompt": {"pool": {"band": list(a.pool_band)}}}))
    if a.corpus:
        base = OmegaConf.merge(base, OmegaConf.create({"prompt": {"pool": None}}))
    train = OmegaConf.load("conf/generated/training_covomix2_eval.yaml")
    for K in (2, 3, 4):
        for suffix, sel in (
            ("", dict(per_session_cap=None, num_windows=None)),
            ("_sub20", dict(per_session_cap=1, num_windows=20, seed=1)),
        ):
            cfg = OmegaConf.merge(base, OmegaConf.create({"selection": dict(num_active_speakers=K, **sel)}))
            t = time.time()
            header, rows = build_eval_manifest(cfg, training_config=train)
            out = Path(f"data/eval/ami_test_k{K}{suffix}_{a.tag}.jsonl")
            write_eval_manifest(out, header, rows)
            same = None
            if a.compare:
                ref = Path(f"data/eval/ami_test_k{K}{suffix}_{a.compare}.jsonl")
                same = window_ids(ref) == window_ids(out) if ref.exists() else None
            pool = header.get("prompt_pool")
            hours = sum(r["t1"] - r["t0"] for r in rows) / 3600
            print(
                f"K={K}{suffix or ' full'}: {len(rows)} windows ({hours:.2f} h), skipped {header['num_skipped']}, "
                + (f"pool {pool['num_speakers']} spk / {pool['num_items']} utts, " if pool else "corpus prompts, ")
                + f"windows identical to {a.compare}: {same}, {time.time() - t:.0f}s -> {out}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
