"""Pool the metrics of sharded infer+measure runs into one summary.

Each shard directory (``<run>/shard<i>/<test_name>/``) carries its own
``scoring/conversation_asr/windows.jsonl`` and the run's ``metrics.json``
one level up (``<run>/shard<i>/metrics.json``).  WER is re-derived from the
per-window, per-channel I/D/S counts pooled over every shard (Thanapat's
rule: sum error counts, never average per-window WERs); the other summary
values are pooled as window-count-weighted means of the shard summaries,
which is exact for per-window means and a close approximation for the
per-minute interaction rates (shards hold near-equal audio).

Usage:
    python local/pool_shard_metrics.py <run_dir> [--test-name valid] [--out <run_dir>/metrics_pooled.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pool_wer(shards: list[Path], test_name: str) -> dict:
    ch = {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    mix = dict(ch)
    n_win = 0
    for sh in shards:
        p = sh / test_name / "scoring" / "conversation_asr" / "windows.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            w = json.loads(line)
            n_win += 1
            for c in w.get("channels", []):
                for k in ch:
                    ch[k] += int(c["counts"].get(k, 0))
            m = w.get("mix") or w.get("mixture")
            if isinstance(m, dict) and "counts" in m:
                for k in mix:
                    mix[k] += int(m["counts"].get(k, 0))

    def wer(c):
        n = c["hits"] + c["substitutions"] + c["deletions"]
        return (c["substitutions"] + c["deletions"] + c["insertions"]) / n if n else None

    return {"n_windows": n_win, "wer_channel": wer(ch), "wer_mix": wer(mix) if any(mix.values()) else None,
            "channel_counts": ch, "mix_counts": mix}


def pool_summaries(shards: list[Path]) -> dict:
    """Window-count-weighted mean of every scalar in the shard metrics.json files."""
    acc: dict[str, float] = {}
    weight: dict[str, float] = {}
    n_shards = 0
    for sh in shards:
        p = sh / "metrics.json"
        if not p.is_file():
            continue
        n_shards += 1
        m = json.loads(p.read_text())
        flat: dict[str, float] = {}
        for v in m.values():
            for vv in v.values():
                flat.update({k: x for k, x in vv.items() if isinstance(x, (int, float))})
        # weight by the shard's scored windows (from windows.jsonl), else 1
        nw = 0
        for wj in sh.glob("*/scoring/conversation_asr/windows.jsonl"):
            nw += sum(1 for l in wj.read_text().splitlines() if l.strip())
        wgt = float(nw or 1)
        for k, x in flat.items():
            acc[k] = acc.get(k, 0.0) + x * wgt
            weight[k] = weight.get(k, 0.0) + wgt
    return {"n_shards": n_shards, **{k: acc[k] / weight[k] for k in acc}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--test-name", default="valid")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    shards = sorted(p for p in a.run_dir.glob("shard*") if p.is_dir())
    if not shards:
        raise SystemExit(f"no shard*/ under {a.run_dir}")
    pooled = pool_summaries(shards)
    wer = pool_wer(shards, a.test_name)
    pooled.update({"wer_channel": wer["wer_channel"], "wer_mix": wer["wer_mix"] if wer["wer_mix"] is not None else pooled.get("wer_mix")})
    result = {"run_dir": str(a.run_dir), "shards": [p.name for p in shards], "n_windows": wer["n_windows"],
              "pooled": pooled, "asr_counts": {"channel": wer["channel_counts"], "mix": wer["mix_counts"]}}
    out = a.out or (a.run_dir / "metrics_pooled.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items()
                      if k in ("n_shards", "wer_channel", "wer_mix", "utmos_ipu_mean", "sim_o_mean", "overlap_sec_per_min", "gap_sec_per_min")}))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
